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
    float x = region_xy.read(FID_X, *pir);
    float y = region_xy.read(FID_Y, *pir);
    region_z.write(FID_Z, *pir, x * alpha + y);
  }
}

void init_value(context _c, rw_region<float, 1> region_xy){
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
    float value = 2;
    region_xy.write(*pir, value);
   // region_xy.write(*pir, FID_Y, y);
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
  
  for (rw_region<float, 1>::iterator pir(region_xy.domain); pir(); pir++) {
    float x = region_xy.read(FID_X, *pir);
    float y = region_xy.read(FID_Y, *pir);
    float z = region_z.read(FID_Z, *pir);
    printf("x %f, y %f, z %f\n", x, y, z);
  }
}

void top_level(context _c)
{ 
  IdxSpace<1> ispace(_c, 10);
  
  FdSpace input_fs(_c);
  input_fs.add_field<float>(FID_X);
  input_fs.add_field<float>(FID_Y);
  Region<1> input_lr(_c, ispace, input_fs);
  
  FdSpace output_fs(_c);
  output_fs.add_field<float>(FID_Z);
  Region<1> output_lr(_c, ispace, output_fs);
  
  std::vector<field_id_t> x_vec{FID_X};
  auto rw_x = rw_region<float, 1>(input_lr, x_vec);
  runtime.execute_task(init_value, _c, rw_x);
  
  std::vector<field_id_t> y_vec{FID_Y};
  auto rw_y = rw_region<float, 1>(input_lr, y_vec);
  runtime.execute_task(init_value, _c, rw_y);
  
  float alpha = 2;
  auto rw_xy = rw_region<float, 1>(input_lr);
  auto rw_z = rw_region<float, 1>(output_lr);
  runtime.execute_task(saxpy, _c, alpha, rw_xy, rw_z);
  
  runtime.execute_task(check, _c, rw_xy, rw_z);
  
 // call((print<float,1>), r);
}

int main(int argc, char** argv){
  runtime.register_task<decltype(&top_level), top_level>("top_level");
  runtime.register_task<decltype(&saxpy), saxpy>("saxpy");
  runtime.register_task<decltype(&check), check>("check");
  runtime.register_task<decltype(&init_value), init_value>("init_value");
  runtime.start(top_level, argc, argv);
}
