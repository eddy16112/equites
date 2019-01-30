#include <legion_simplified.h>
using namespace LegionSimplified;

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
  /*
  RO_Region<1>::iterator pir;
  for(pir = region_xy.begin(); pir != region_xy.end(); pir++) {
    float x = region_xy.read<float>(FID_X, *pir);
    float y = region_xy.read<float>(FID_Y, *pir);
    region_z.write<float>(FID_Z, *pir, x * alpha + y);
  }*/
  /*
  for (RO_Region<1>::iterator pir(region_xy); pir(); pir++) {
    float x = region_xy.read<float>(FID_X, *pir);
    float y = region_xy.read<float>(FID_Y, *pir);
    region_z.write<float>(FID_Z, *pir, x * alpha + y);
  }*/
}

void init_value(context c, WD_Region<1> region_xy){
  /*
  for (WD_Region<1>::iterator pir(region_xy); pir(); pir++) {
    float value = 2;
    region_xy.write<float>(*pir, value);
  }
  WD_Region<1> tmp = region_xy;
  int id = 0;
  auto pir = region_xy.begin();
  for(pir = region_xy.begin(); pir != region_xy.end(); pir++) {
    float value = 2;
    printf("id %d\n", id);
    id ++;
    region_xy.write<float>(*pir, value);
  }
  */
  
  for(auto pir : region_xy) {
    float value = 2;
    region_xy.write<float>(pir, value);
  }
  
}

void check(context c, float alpha, RO_Region<1> region_xy, RO_Region<1> region_z){
  
  for(auto pir : region_xy) {
    float x = region_xy.read<float>(FID_X, pir);
    float y = region_xy.read<float>(FID_Y, pir);
    float z = region_z.read<float>(FID_Z, pir);
    if (z != x*alpha + y) {
      printf("failed\n");
    }
  } 
  printf("success!!!\n");
  
  
  /*
  for (RW_Region<1>::iterator pir(region_xy); pir(); pir++) {
    float x = region_xy.read<float>(FID_X, *pir);
    float y = region_xy.read<float>(FID_Y, *pir);
    float z = region_z.read<float>(FID_Z, *pir);
    printf("x %f, y %f, z %f\n", x, y, z);
  }
  */
  
}

void top_level(context c)
{ 
  IdxSpace<1> ispace(c, 10);
  
  FdSpace input_fs(c);
  input_fs.add_field<float>(FID_X);
  input_fs.add_field<float>(FID_Y);
  Region<1> input_lr(ispace, input_fs);
  
  FdSpace output_fs(c);
  output_fs.add_field<float>(FID_Z);
  Region<1> output_lr(ispace, output_fs);
  
  std::vector<field_id_t> x_vec{FID_X};
  auto wd_x = WD_Region<1>(input_lr, x_vec);
  //auto tmp = wd_x;
  //printf("count %ld, tmp count %d\n", wd_x.base_region_impl.use_count(), tmp.base_region_impl.use_count());
  runtime.execute_task(init_value, c, wd_x);
  
  std::vector<field_id_t> y_vec{FID_Y};
  auto wd_y = WD_Region<1>(input_lr, y_vec);
  runtime.execute_task(init_value, c, wd_y);
  
  float alpha = 2;
  auto rw_xy = RO_Region<1>(input_lr);
  auto wd_z = WD_Region<1>(output_lr);
  runtime.execute_task(saxpy, c, alpha, rw_xy, wd_z);
  
  auto ro_xy = RO_Region<1>(input_lr);
  auto ro_z = RO_Region<1>(output_lr);
  runtime.execute_task(check, c, alpha, ro_xy, ro_z);
 
 // call((print<float,1>), r);
}

int main(int argc, char** argv){
  runtime.register_task<decltype(&top_level), top_level>("top_level");
  runtime.register_task<decltype(&saxpy), saxpy>("saxpy", true);
  runtime.register_task<decltype(&check), check>("check", true);
  runtime.register_task<decltype(&init_value), init_value>("init_value", true);
  runtime.start(top_level, argc, argv);
}
