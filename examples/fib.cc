#include <equites.h>
using namespace equites;

task(int, fib, int i){
  if(i < 2) return 1; 
  auto x = call(fib, i-1);
  auto y = call(fib, i-2);
  return y.get() + x.get();
}

task(void, toplevel, int argc, char** argv){
  int n  = argc > 1 ? atoi(argv[1]) : 15; 
  printf("fib(%d) = %d\n", n, call(fib, n).get()); 
}

int main(int argc, char** argv){
  start(toplevel, argc, argv);
}
