#include <equites.h>
using namespace equites;

task(int, fib, int i){
  if(i < 2) return 1; 
  auto x = call(fib, i-1);
  auto y = call(fib, i-2);
  return x.get() + y.get();
}

task(void, toplevel, int argc, char** argv){
  int n = 15; 
  if(argc > 2) n = atoi(argv[1]); 
  printf("fib(%d) = %d\n", n, call(fib, n).get()); 
}

int main(int argc, char** argv){
  start(toplevel, argc, argv);
}
