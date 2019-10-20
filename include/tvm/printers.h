/*!
 * \file tvm/printers.h
 * \brief Printer functions for tvm classes
 */
#ifndef TVM_PRINTERS_H_
#define TVM_PRINTERS_H_

#include <string>
#include <vector>
#include <iostream>

#include <tvm/operation.h>
namespace tvm {

/** Print vector of object in human-readable form */
template<typename T>
extern std::ostream &operator<<(std::ostream &stream, const std::vector<T> &);

/** Print vector of vector of object in human-readable form */
template<typename T>
extern std::ostream &operator<<(std::ostream &stream, const std::vector<std::vector<T> > &);

extern std::ostream& operator<< (std::ostream &out, const TensorDom& tdom);


} //namespace tvm
#endif //TVM_PRINTERS_H_
