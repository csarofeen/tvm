#include <tvm/printers.h>

namespace tvm{

template<typename T>
std::ostream &operator<<(std::ostream &stream, const std::vector<T> &vec){
  stream<<"{";
  for(int i=0; i<vec.size(); ++i){
    stream<<vec[i];
    if(i != vec.size()-1)
      stream<<", ";
  }
  stream<<"}";
  return stream;
}

template<typename T>
std::ostream &operator<<(std::ostream &stream, const std::vector<std::vector<T> > &vecovec){
  stream<<"{\n";
  for(int i=0; i<vecovec.size(); ++i){
    stream<<"\t"<<vecovec[i];
    if(i != vecovec.size()-1)
      stream<<",\n";
  }
  stream<<"}";
  return stream;
}

extern std::ostream& operator<< (std::ostream &stream, const TensorDom& tdom){

  stream<<"TDom ["<<tdom.data.size()<<"] : \n";
  stream<<tdom.data;
  return stream;
  /*
  for(uint i=0; i<tdom.data.size(); ++i){
    out<<"\t{";
    for(uint j=0; j<tdom.data[i].size(); ++j){
      out<<tdom.data[i][j];
      if(j!=tdom.data[i].size()-1)
	out<<", ";
    }
    out<<"}";
    if(i!=tdom.data.size()-1)
      out<<"\n";
  }
  return out;
  */
}

}
