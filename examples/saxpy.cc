#include <equites.h>
using namespace equites;

enum FieldIDs {
  FID_X = 3,
  FID_Y,
  FID_Z,
};

void saxpy(context _c, float alpha, rw_region<float, 1> region_xy, rw_region<float, 1> region_z){
  /*
  for(auto i : ab) {
    float z = c.read(i, FID_Z);
    float x = ab.read(i, FID_X);
    float y = ab.read(i, FID_Y);
    c.write(i, FID_Z, z + x * alpha + y);
  }*/
  for (rw_region<float, 1>::iterator pir(region_xy.domain); pir(); pir++) {
    float z = region_z.read(*pir, FID_Z);
    float x = region_xy.read(*pir, FID_X);
    float y = region_xy.read(*pir, FID_Y);
    region_z.write(*pir, FID_Z, z + x * alpha + y);
  }
}

void init_value(context _c, rw_region<float, 1> region_xy, rw_region<float, 1> region_z){
  /*
  for(auto i : ab) {
    float z = 3;
    float x = 1;
    float y = 2;
    ab.write(i, FID_X, x);
    ab.write(i, FID_Y, y);
    c.write(i, FID_Z, z);
  }*/
  
  for (rw_region<float, 1>::iterator pir(region_xy.domain); pir(); pir++) {
    float x = 1;
    float y = 2;
    float z = 3;
    region_xy.write(*pir, FID_X, x);
    region_xy.write(*pir, FID_Y, y);
    region_z.write(*pir, FID_Z, z);
  }
}

void check(context _c, rw_region<float, 1> region_xy, rw_region<float, 1> region_z){
  /*
  for(auto i : ab) {
    float z = c.read(i, FID_Z);
    float x = ab.read(i, FID_X);
    float y = ab.read(i, FID_Y);
    printf("x %f, y %f, z %f\n", x, y, z);
  } */
  
 // Legion::PointInDomainIterator<1> pir(ab.domain);
  for (rw_region<float, 1>::iterator pir(region_xy.domain); pir(); pir++) {
    float x = region_xy.read(*pir, FID_X);
    float y = region_xy.read(*pir, FID_Y);
    float z = region_z.read(*pir, FID_Z);
    printf("x %f, y %f, z %f\n", x, y, z);
  }
}

void top_level(context _c)
{
 // auto r = region(float, 1, 4);

 // call((fill<float,1>), r, 3); 
  
  IdxSpace<1> ispace(_c, 4);
  FdSpace input_fs(_c);
  input_fs.add_field<float>(FID_X);
  input_fs.add_field<float>(FID_Y);
  auto input_lr = rw_region<float, 1>(_c, ispace, input_fs);
  input_lr.set_task_field(FID_X);
  input_lr.set_task_field(FID_Y);
  
  FdSpace output_fs(_c);
  output_fs.add_field<float>(FID_Z);
  auto output_lr = rw_region<float, 1>(_c, ispace, output_fs);
  output_lr.set_task_field(FID_Z);
  
  //auto input_r = rw_region<float,1>(_c, make_point(4));
  runtime.execute_task(init_value, _c, input_lr, output_lr);
  
  float alpha = 2;
  runtime.execute_task(saxpy, _c, alpha, input_lr, output_lr);
  
  runtime.execute_task(check, _c, input_lr, output_lr);
  
 // call((print<float,1>), r);
}

int main(int argc, char** argv){
  runtime.register_task<decltype(&top_level), top_level>("top_level");
  runtime.register_task<decltype(&saxpy), saxpy>("saxpy");
  runtime.register_task<decltype(&check), check>("check");
  runtime.register_task<decltype(&init_value), init_value>("init_value");
  runtime.start(top_level, argc, argv);
}
