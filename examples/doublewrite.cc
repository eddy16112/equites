#include <equites.h>
using namespace equites;

task(void, toplevel){
  auto r = region(float, 1, 4);
  call((fill<float, 1>), r, 3.14159); 
  call((fill<float, 1>), r, 6.28319); 
  call((print<float, 1>), r);
}

int main(int argc, char** argv){
  start(toplevel, argc, argv);
}
