#include <equites.h>
using namespace equites;

void hello(context c, int i)
{
  printf("hello world %d\n", i);
}

void top_level(context c)
{
  printf("top level\n");
  runtime.execute_task(hello, c, 2);
  runtime.execute_task(hello, c, 3);
  runtime.execute_task(hello, c, 4);
  runtime.execute_task(hello, c, 5);
}

int main(int argc, char** argv){
  runtime.register_task<decltype(&top_level), top_level>("top_level");
  runtime.register_task<decltype(&hello), hello>("hello");
  runtime.start(top_level, argc, argv);
}

