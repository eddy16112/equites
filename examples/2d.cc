#include <equites.h>
using namespace equites;

task(void, toplevel){
  auto r1 = region(float, 2, make_point(2, 2));
  auto r2 = region(float, 2, make_point(2, 2));
  call((fill<float,2>), r1, 3.14159); 
  call((copy<float,2>), r1, r2);
  std::cout << "r1" << std::endl; 
  call((print<float,2>), r2).get();
  std::cout << "r2" << std::endl; 
  call((print<float,2>), r1); 
}

int main(int argc, char** argv){
  start(toplevel, argc, argv);
}
