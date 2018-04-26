#include <equites.h>
using namespace equites;

void inc(context _c, rw_region<int, 1> r, int x) {
  for(auto i : r)
    r.write(i, r.read(i) + x); 
}

void top_level(context _c)
{
  auto r = region(int, 1, 4); 
  call((fill<int,1>), r, 0);
  runtime.execute_task(inc, _c, r, 5);
  call((print<int,1>), r); 
}

int main(int argc, char** argv){
  runtime.register_task<decltype(&top_level), top_level>("top_level");
  runtime.register_task<decltype(&inc), inc>("inc");
  runtime.start(top_level, argc, argv);
}

