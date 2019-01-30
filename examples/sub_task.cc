#include <legion_simplified.h>
using namespace LegionSimplified;

enum FieldIDs {
  FID_X = 3,
  FID_Y,
};

void init_value(context c, WD_Region<1> region_xy){
  
  for(auto pir : region_xy) {
    float value = 2;
    region_xy.write<float>(FID_X, pir, value);
    region_xy.write<float>(FID_Y, pir, value);
  }
  /*
  for (WD_Region<2>::iterator pir(region_xy); pir(); pir++) {
    float value = 2;
    region_xy.write<float>(FID_X, *pir, value);
    region_xy.write<float>(FID_Y, *pir, value);
  }*/
}

void print_value_2(context c, RW_Region<1> region_xy){
  /*
  for(auto i : ab) {
    float z = c.read(i, FID_Z);
    float x = ab.read(i, FID_X);
    float y = ab.read(i, FID_Y);
    printf("x %f, y %f, z %f\n", x, y, z);
  } */
  
  for (RO_Region<1>::iterator pir(region_xy); pir(); pir++) {
    float x = region_xy.read<float>(FID_X, *pir);
    double y = region_xy.read<float>(FID_Y, *pir);
    printf("[2] x %f, y %f\n", x, y);
  }
  
  printf("end print value 2\n");
}

void print_value(context c, RW_Region<1> region_xy){
  /*
  for(auto i : ab) {
    float z = c.read(i, FID_Z);
    float x = ab.read(i, FID_X);
    float y = ab.read(i, FID_Y);
    printf("x %f, y %f, z %f\n", x, y, z);
  } */
  
  float value = 3;
  for (RO_Region<1>::iterator pir(region_xy); pir(); pir++) {
    float x = region_xy.read<float>(FID_X, *pir);
    double y = region_xy.read<float>(FID_Y, *pir);
    printf("x %f, y %f\n", x, y);
    
    region_xy.write<float>(FID_X, *pir, value);
    region_xy.write<float>(FID_Y, *pir, value);
  }
  
  Region<1> input_lr = region_xy.get_region();
  IdxSpace<1> color_is(c, 2);
  Partition<1> input_lp(equal, input_lr, color_is);
  
  auto ro_xy = RW_Region<1>(input_lp);
  runtime.execute_task(print_value_2, c, color_is, ro_xy);
  
  printf("end print value\n");
}

void top_level(context c)
{ 
  IdxSpace<1> ispace(c, 12);
  
  FdSpace input_fs(c);
  input_fs.add_field<float>(FID_X);
  input_fs.add_field<float>(FID_Y);
  Region<1> input_lr(ispace, input_fs);
  
  auto wd_xy = WD_Region<1>(input_lr);
  runtime.execute_task(init_value, c, wd_xy);
  
  auto ro_xy = RW_Region<1>(input_lr);
  runtime.execute_task(print_value, c, ro_xy);
  
  printf("end top level\n");
}

int main(int argc, char** argv){
  runtime.register_task<decltype(&top_level), top_level>("top_level");
  runtime.register_task<decltype(&init_value), init_value>("init_value", true);
  runtime.register_task<decltype(&print_value), print_value>("print_value");
  runtime.register_task<decltype(&print_value_2), print_value_2>("print_value_2", true);
  runtime.start(top_level, argc, argv);
}
