#include <equites.h>
using namespace equites;

enum FieldIDs {
  FID_X = 3,
  FID_Y,
  FID_Z,
};

void saxpy(context c, float alpha, RO_Region<1> region_xy, WD_Region<1> region_z){
  for(auto pir : region_xy) {
    float x = region_xy.read<float>(FID_X, pir);
    float y = region_xy.read<float>(FID_Y, pir);
    region_z.write<float>(FID_Z, pir, x * alpha + y);
  }
  
  const int point = c.task->index_point.point_data[0];
  /*
  for (RO_Region<1>::iterator pir(region_xy); pir(); pir++) {
    float x = region_xy.read<float>(FID_X, *pir);
    float y = region_xy.read<float>(FID_Y, *pir);
    region_z.write<float>(FID_Z, *pir, x * alpha + y);
  }*/
  printf("saxpy point %d\n", point);
}

void init_value(context c, WD_Region<1> region_xy){
  for (WD_Region<1>::iterator pir(region_xy); pir(); pir++) {
    float value = 2;
    region_xy.write<float>(*pir, value);
  }
}

void check(context c, RO_Region<1> region_xy, RO_Region<1> region_z){
  /*
  for(auto i : ab) {
    float z = c.read(i, FID_Z);
    float x = ab.read(i, FID_X);
    float y = ab.read(i, FID_Y);
    printf("x %f, y %f, z %f\n", x, y, z);
  } */
  
  for (RW_Region<1>::iterator pir(region_xy); pir(); pir++) {
    float x = region_xy.read<float>(FID_X, *pir);
    float y = region_xy.read<float>(FID_Y, *pir);
    float z = region_z.read<float>(FID_Z, *pir);
    printf("x %f, y %f, z %f\n", x, y, z);
  }
}

void top_level(context c)
{ 
  IdxSpace<1> ispace(c, 12);
  IdxSpace<1> color_is(c, 4);
  
  FdSpace input_fs(c);
  input_fs.add_field<float>(FID_X);
  input_fs.add_field<float>(FID_Y);
  Region<1> input_lr(c, ispace, input_fs);
  
  FdSpace output_fs(c);
  output_fs.add_field<float>(FID_Z);
  Region<1> output_lr(c, ispace, output_fs);
  
  Partition<1> input_lp(c, equal, input_lr, color_is);
  Partition<1> output_lp(c, equal, output_lr, color_is);
  
  std::vector<field_id_t> x_vec{FID_X};
  auto wd_x = WD_Region<1>(&input_lp, x_vec);
  runtime.execute_task(init_value, c, color_is, wd_x);
  
  std::vector<field_id_t> y_vec{FID_Y};
  auto wd_y = WD_Region<1>(&input_lp, y_vec);
  runtime.execute_task(init_value, c, color_is, wd_y);
  
  float alpha = 2;
  auto rw_xy = RO_Region<1>(&input_lp);
  auto wd_z = WD_Region<1>(&output_lp);
  runtime.execute_task(saxpy, c, color_is, alpha, rw_xy, wd_z);
  
  auto ro_xy_all = RO_Region<1>(&input_lr);
  auto ro_z_all = RO_Region<1>(&output_lr);
  runtime.execute_task(check, c, ro_xy_all, ro_z_all);
  
 // call((print<float,1>), r);
}

int main(int argc, char** argv){
  runtime.register_task<decltype(&top_level), top_level>("top_level");
  runtime.register_task<decltype(&saxpy), saxpy>("saxpy");
  runtime.register_task<decltype(&check), check>("check");
  runtime.register_task<decltype(&init_value), init_value>("init_value");
  runtime.start(top_level, argc, argv);
}
