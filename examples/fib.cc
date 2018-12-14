#include <legion_simplified.h>
using namespace LegionSimplified;

int fib(context c, int i){
  if(i < 2) return 1; 
  Future x = runtime.execute_task(fib, c, i-1);
  Future y = runtime.execute_task(fib, c, i-2);
  return y.get<int>() + x.get<int>();
}

void top_level(context c, int argc, char** argv){
  int n  = argc > 1 ? atoi(argv[1]) : 15; 
  printf("fib(%d) = %d\n", n, runtime.execute_task(fib, c, n).get<int>()); 
}

int main(int argc, char** argv){
  runtime.register_task<decltype(&top_level), top_level>("top_level");
  runtime.register_task<decltype(&fib), fib>("fib");
  runtime.start(top_level, argc, argv);
}
