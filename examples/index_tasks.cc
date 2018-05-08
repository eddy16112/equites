#include <equites.h>
using namespace equites;

int hello(context c)
{
  int i = ArgMap::get_arg<int>(c);
  printf("hello world %d\n", i);
  return i;
}

void top_level(context c)
{
  printf("top level\n");
  int num_points = 4;
  IdxSpace<1> ispace(c, num_points);
  ArgMap arg_map;
  for (int i = 0; i < num_points; i++) {
    arg_map.set_point<int>(i, i);
  }
  FutureMap fm = runtime.execute_task(hello, c, ispace, arg_map);
  fm.wait();
  for (int i = 0; i < num_points; i++) {
    int received = fm.get<int>(i);
    printf("received %d, expected %d\n", received, i);
  }
}

int main(int argc, char** argv){
  runtime.register_task<decltype(&top_level), top_level>("top_level");
  runtime.register_task<decltype(&hello), hello>("hello");
  runtime.start(top_level, argc, argv);
}

