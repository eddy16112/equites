#include <equites.h>
using namespace equites;

void top_level(context _c)
{
 // auto r = region(float, 1, 4);
  auto r = rw_region<float,1>(_c, make_point(4));
  IdxSpace<1> ispace(_c, 3);
  Legion::Point<1> pt(2);
  pt.x = 3;
  call((fill<float,1>), r, 3.14); 
  call((print<float,1>), r);
}

int main(int argc, char** argv){
  runtime.register_task<decltype(&top_level), top_level>("top_level");
  runtime.start(top_level, argc, argv);
}
