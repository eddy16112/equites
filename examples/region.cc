#include <equites.h>
using namespace equites;

task(void, print, r_region<float, 1> r){
  for(auto i : r){
    printf("%f\n", r.read(i));
  }
}

task(void, init, w_region<float, 1> r){
  for(Point<1> i : r){
    r.write(i, 3.14159);  
  }
}

task(void, toplevel){
  auto r = region(float, 1, 4);
  call(init, r); 
  call(print, r);
}

int main(int argc, char** argv){
  start(toplevel, argc, argv);
}
