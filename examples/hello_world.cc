#include <legion_simplified.h>
using namespace LegionSimplified;

void hello_world(context c, int i)
{
  printf("hello world %d\n", i);
}

void top_level(context c)
{
  printf("top level\n");
  runtime.execute_task(hello_world, c, 1);
  runtime.execute_task(hello_world, c, 2);
  runtime.execute_task(hello_world, c, 3);
  runtime.execute_task(hello_world, c, 4);
}

int main(int argc, char** argv){
  runtime.register_task<decltype(&top_level), top_level>("top_level");
  runtime.register_task<decltype(&hello_world), hello_world>("hello_world");
  runtime.start(top_level, argc, argv);
}

