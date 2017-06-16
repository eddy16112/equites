#include <equites.h>
using namespace equites;

task(void, printFirst, r_region<float, 1> r){
  printf("%f\n", r.read(0));
}

task(void, init, w_region<float, 1> r){
  r.write(0, 3.14159);
}

task(void, toplevel){
  auto r = region(float, 1, 1);
  call(init, r); 
  call(printFirst, r);
}

int main(int argc, char** argv){
  start(toplevel, argc, argv);
}
