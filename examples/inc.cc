#include <equites.h>
using namespace equites;

task(void, inc, rw_region<int, 1> r, int x) {
  for(auto i : r)
    r.write(i, r.read(i) + x); 
}

task(void, maintask){
  auto r = region(int, 1, 4); 
  call((fill<int,1>), r, 0);
  call(inc, r, 2); 
  call((print<int,1>), r); 
}

int main(int argc, char** argv){
  start(maintask, argc, argv);
}

