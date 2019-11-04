/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2017 by Contributors
 * \file make_api.cc Build API function.
 */
#include <tvm/ir_pass.h>
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/buffer.h>
#include <tvm/runtime/device_api.h>
#include <vector>
#include <utility>
#include <unordered_set>
#include <stack>
#include "ir_util.h"
#include "arg_binder.h"

namespace tvm {
namespace ir {

/*
There are instances where the bounds pass will infer the shape of a previous
stage to be based upon the ThreadIdx.x. This can look like:

// attr [compute(C, 0x55e4e813c440)] realize_scope = ""

realize C([0, 1024], [0, 1024]) {
  produce C {
    // attr [iter_var(blockIdx.x, , blockIdx.x)] thread_extent = 1024
    // attr [compute(B, 0x55e4e811b350)] realize_scope = "local"
    realize B([blockIdx.x, 1], [threadIdx.x, 1]) {
      produce B {
        B(blockIdx.x, threadIdx.x) =(A(blockIdx.x, threadIdx.x)*1.001f)
      }
      // attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 1024
      C(blockIdx.x, threadIdx.x) =(B(blockIdx.x, threadIdx.x)*2f)
    }
  }
}

The issue in an example like this is
// attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 1024
comes after the use in "produce B".

This pass will check for use of blockIdx or threadIdx and will lift the
attr statement to the begining of the producer that uses it.
*/

class LiftThreads : public IRMutator {

  //Keep track of thread definitions to see if we're missing one
  std::unordered_map<const Variable*, int> defined_threads;
  //Translation from thread variable to its associated attr stmt
  std::unordered_map<const Variable*, Stmt> thread_var2stmt;
  //Keep track of variables that weren't defined but that were used
  std::stack<std::set<const Variable*> > needed_vars;
  int stack_level = 0;

  //Clear out the thread definitions found in the current scope
  //as we're leaving that scope
  void clear_definitions(){
    std::unordered_map<const Variable*, int> new_definition;
    for(auto it = defined_threads.begin(); it != defined_threads.end(); ++it){
      if(it->second != stack_level){
	new_definition.emplace(it->first, it->second);
      }
    }
    defined_threads = new_definition;
  }

public:

  Expr Mutate_(const Variable *op, const Expr& e) {

    bool is_thread = false;
    std::string name = op->name_hint;

    if (name.compare(0, 9, "blockIdx.") == 0
	&& name.length() == 10
	&& static_cast<int>(name[9] - 'x') < 3
	) is_thread = true;

    else if (name.compare(0, 10, "threadIdx.") == 0
	&& name.length() == 11
	&& static_cast<int>(name[10] - 'x') < 3
	) is_thread =  true;

    if(is_thread)
      if(defined_threads.count(op) == 0)
	needed_vars.top().emplace(op);

    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt& s) {
    if (op->attr_key != attr::thread_extent)
      return IRMutator::Mutate_(op, s);

    IterVar iv = Downcast<IterVar>(op->node);
    const Variable *var = iv->var.get();
    defined_threads.emplace(var, stack_level);
    if(thread_var2stmt.count(iv->var.get()) == 0){
      Stmt thread_stmt =
	AttrStmt::make(iv, "thread_extent", Expr(op->value), Evaluate::make(0));
      thread_var2stmt.emplace(iv->var.get(), thread_stmt);
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const ProducerConsumer *op, const Stmt& s){
    if(!op->is_producer)
      return IRMutator::Mutate_(op, s);
    ++stack_level;

    needed_vars.push(std::set<const Variable*>());
    auto body = IRMutator::Mutate(op->body);
    if(needed_vars.top().size() != 0){
      for(const auto& var : needed_vars.top()){
	if(thread_var2stmt.find(var) == thread_var2stmt.end())
	  continue;

	const AttrStmt* needed_attr = thread_var2stmt.find(var)->second.as<AttrStmt>();
	body = AttrStmt::make(needed_attr->node,
			      needed_attr->attr_key, needed_attr->value, body);
      }
    }


    needed_vars.pop();
    clear_definitions();

    //Final result, top level producer
    if(stack_level == 1){
      needed_vars.push(std::set<const Variable*>());
      auto body = IRMutator::Mutate_(op, s);
      needed_vars.pop();
      clear_definitions();
      --stack_level;
      return body;
    }

    --stack_level;
    return ProducerConsumer::make(op->func, op->is_producer, body);
  }

};

inline Stmt MakeAssertEQ(Expr lhs, Expr rhs, std::string msg) {
  return AssertStmt::make(lhs == rhs, msg, Evaluate::make(0));
}

LoweredFunc MakeAPI(Stmt body,
                    std::string name,
                    Array<NodeRef> api_args,
                    int num_unpacked_args,
                    bool is_restricted) {
  const Stmt nop = Evaluate::make(0);
  int num_args = static_cast<int>(api_args.size());
  CHECK_LE(num_unpacked_args, num_args);
  int num_packed_args = num_args - num_unpacked_args;
  // Data field definitions
  // The packed fields
  Var v_packed_args("args", Handle());
  Var v_packed_arg_type_ids("arg_type_ids", Handle());
  Var v_num_packed_args("num_args", Int(32));
  // The arguments of the function.
  Array<Var> args;
  // The device context
  Var device_type("dev_type"), device_id("dev_id");
  // seq_init gives sequence of initialization
  // seq_check gives sequence of later checks after iniit
  std::vector<Stmt> seq_init, seq_check;
  std::unordered_map<const Variable*, Expr> vmap;
  ArgBinder binder(&vmap);
  // ---------------------------
  // local function defintiions
  // load i-th argument as type t
  auto f_arg_value = [&](Type t, int i) {
    Array<Expr> call_args{v_packed_args,
                          IntImm::make(Int(32), i),
                          IntImm::make(Int(32), intrinsic::kTVMValueContent)};
    // load 64 bit version
    Type api_type = APIType(t);
    Expr res = Call::make(
        api_type, intrinsic::tvm_struct_get, call_args,
        Call::PureIntrinsic);
    // cast to the target version.
    if (api_type != t) {
      res = Cast::make(t, res);
    }
    return res;
  };
  // get declaration of argument i
  auto f_arg_decl = [&](int i) {
    std::ostringstream os;
    os << "arg" << i;
    const Variable* v = api_args[i].as<Variable>();
    return Var(os.str(), v ? v->type: Handle());
  };
  // ---------------------------
  // start of logics
  // add signiture for packed arguments.
  if (num_packed_args != 0) {
    args.push_back(v_packed_args);
    args.push_back(v_packed_arg_type_ids);
    args.push_back(v_num_packed_args);
    std::ostringstream os;

    os << name << ": num_args should be " << num_packed_args;
    seq_init.emplace_back(
        MakeAssertEQ(v_num_packed_args, num_packed_args, os.str()));
  }

  // Save the input variables and buffers that will be bound later.
  std::vector<std::pair<Var, Var> > var_defs;
  std::vector<std::pair<Buffer, Var> > buf_defs;
  for (int i = 0; i < static_cast<int>(api_args.size()); ++i) {
    Var v_arg = f_arg_decl(i);
    if (i < num_packed_args) {
      // Value loads
      seq_init.emplace_back(LetStmt::make(
          v_arg, f_arg_value(v_arg.type(), i), nop));
      // type code checks
      Var tcode(v_arg->name_hint + ".code", Int(32));
      seq_init.emplace_back(LetStmt::make(
          tcode, Load::make(
              Int(32), v_packed_arg_type_ids, IntImm::make(Int(32), i), const_true(1)),
          nop));
      Type t = v_arg.type();
      if (t.is_handle()) {
        std::ostringstream msg;
        msg << name << ": Expect arg[" << i << "] to be pointer";
        seq_check.emplace_back(
            AssertStmt::make(tcode == kHandle ||
                             tcode == kNDArrayContainer ||
                             tcode == kArrayHandle ||
                             tcode == kNull, msg.str(), nop));
      } else if (t.is_int() || t.is_uint()) {
        std::ostringstream msg;
        msg << name << ": Expect arg[" << i << "] to be int";
        seq_check.emplace_back(AssertStmt::make(tcode == kDLInt, msg.str(), nop));
      } else {
        CHECK(t.is_float());
        std::ostringstream msg;
        msg << name << ": Expect arg[" << i << "] to be float";
        seq_check.emplace_back(
            AssertStmt::make(tcode == kDLFloat, msg.str(), nop));
      }
    } else {
      args.push_back(v_arg);
    }
    // add checks for functions.
    if (api_args[i].as<Variable>()) {
      var_defs.emplace_back(std::make_pair(Downcast<Var>(api_args[i]), v_arg));
    } else {
      // Buffer checks
      CHECK(api_args[i].as<BufferNode>())
          << "api_args can only be Buffer or Var";
      buf_defs.emplace_back(std::make_pair(Downcast<Buffer>(api_args[i]), v_arg));
    }
  }

  // Arg definitions are defined before buffer binding to avoid the use before
  // def errors.
  //
  // For example, for auto broadcasting, checks are required to guarantee that
  // either 0 or the original stride will be correctly used. Checks here have
  // to use the args that may have no let bining yet. Therefore, hoisting let
  // binding for args before buffer declaration is needed.
  for (const auto& arg : var_defs) {
    binder.Bind(arg.first, arg.second, arg.second->name_hint, true);
  }

  for (const auto& buf_arg : buf_defs) {
    binder.BindDLTensor(buf_arg.first, device_type, device_id,
                        buf_arg.second, buf_arg.second->name_hint);
  }

  NodePtr<LoweredFuncNode> n = make_node<LoweredFuncNode>();
  n->name = name;
  n->args = args;
  n->handle_data_type = binder.def_handle_dtype();
  n->is_packed_func = num_unpacked_args == 0;
  n->is_restricted = is_restricted;
  body = AttrStmt::make(
      make_zero(Int(32)), attr::compute_scope,
      StringImm::make(name + "_compute_"), body);
  // Set device context
  if (vmap.count(device_id.get())) {
    Expr node = StringImm::make("default");
    CHECK(vmap.count(device_type.get()));
    seq_check.push_back(AttrStmt::make(
        node, attr::device_context_id, device_id, nop));
    seq_check.push_back(AttrStmt::make(
        node, attr::device_context_type, device_type, nop));
    Stmt set_device = IfThenElse::make(
        device_type != kDLCPU, Evaluate::make(Call::make(
            Int(32), intrinsic::tvm_call_packed,
            {StringImm::make(runtime::symbol::tvm_set_device),
             device_type, device_id}, Call::Intrinsic)));
    body = Block::make(set_device, body);
  }
  n->body = MergeNest(
      {seq_init, binder.init_nest(), seq_check, binder.asserts()}, body);

  body = LiftThreads().Mutate(body);

  LoweredFunc f(n);
  Array<Var> undefined = UndefinedVars(f->body, f->args);
  if (undefined.size() != 0) {
    std::ostringstream os;
    for (Var v : undefined) {
      os << " \'" << v->name_hint << "\' ";
    }
    os << " does not appear in api_args";
    LOG(FATAL) << "Not all Vars are passed in api_args: " << os.str();
  }
  return f;
}

class DeviceTypeBinder: public IRMutator {
 public:
  explicit DeviceTypeBinder(int device_type)
      : device_type_(device_type) {}

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == attr::device_context_type) {
      if (const Variable* var = op->value.as<Variable>()) {
        var_ = var;
        Expr value = make_const(op->value.type(), device_type_);
        Stmt body = IRMutator::Mutate_(op, s);
        var_ = nullptr;
        std::ostringstream os;
        os << "device_type need to be " << device_type_;
        return AssertStmt::make(op->value == value, os.str(), body);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const IfThenElse* op, const Stmt& s) final {
    // eager simplify if guard.
    Stmt res = IRMutator::Mutate_(op, s);
    op = res.as<IfThenElse>();
    if (is_zero(op->condition)) {
      if (op->else_case.defined()) return op->else_case;
      return Evaluate::make(0);
    }
    if (is_one(op->condition)) {
      return op->then_case;
    }
    return res;
  }

  Expr Mutate_(const NE* op, const Expr& e) final {
    // eager check NE for device check
    Expr res = IRMutator::Mutate_(op, e);
    op = res.as<NE>();
    if (ir::Equal(op->a, op->b)) {
      return make_const(op->type, false);
    }
    return res;
  }

  Expr Mutate_(const Variable* op, const Expr& e) final {
    if (op == var_) {
      return make_const(op->type, device_type_);
    } else {
      return e;
    }
  }

 public:
  const Variable* var_{nullptr};
  int device_type_;
};

LoweredFunc BindDeviceType(LoweredFunc f,
                           int device_type) {
  auto n = make_node<LoweredFuncNode>(*f.operator->());
  n->body = DeviceTypeBinder(device_type).Mutate(n->body);
  return LoweredFunc(n);
}

}  // namespace ir
}  // namespace tvm
