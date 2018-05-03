#include <equites.h>
using namespace equites;

void hello(context c, int i)
{
  printf("hello world %d\n", i);
}

void top_level(context c)
{
  printf("top level\n");
  IdxSpace<1> ispace(c, 4);
  runtime.execute_task(hello, c, ispace, 1);
}

int main(int argc, char** argv){
  runtime.register_task<decltype(&top_level), top_level>("top_level");
  runtime.register_task<decltype(&hello), hello>("hello");
  runtime.start(top_level, argc, argv);
}

