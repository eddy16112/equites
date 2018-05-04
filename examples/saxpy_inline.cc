#include <equites.h>
using namespace equites;

enum FieldIDs {
  FID_X = 3,
  FID_Y,
  FID_Z,
};

void saxpy(context c, float alpha, RW_Region<1> region_xy, WD_Region<1> region_z){
  /*
  for(auto i : ab) {
    float z = c.read(i, FID_Z);
    float x = ab.read(i, FID_X);
    float y = ab.read(i, FID_Y);
    c.write(i, FID_Z, z + x * alpha + y);
  }*/
  for (RW_Region<1>::iterator pir(region_xy); pir(); pir++) {
    float x = region_xy.read<float>(FID_X, *pir);
    float y = region_xy.read<float>(FID_Y, *pir);
    region_z.write<float>(FID_Z, *pir, x * alpha + y);
  }
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
  IdxSpace<1> ispace(c, 10);
  
  FdSpace input_fs(c);
  input_fs.add_field<float>(FID_X);
  input_fs.add_field<float>(FID_Y);
  Region<1> input_lr(c, ispace, input_fs);
  
  FdSpace output_fs(c);
  output_fs.add_field<float>(FID_Z);
  Region<1> output_lr(c, ispace, output_fs);
  
  float alpha = 2;
  auto rw_xy = RW_Region<1>(&input_lr);
  auto wd_z = WD_Region<1>(&output_lr);
  
  rw_xy.map_physical_inline(c);
  wd_z.map_physical_inline(c);
  
  for (RW_Region<1>::iterator pir(rw_xy); pir(); pir++) {
    rw_xy.write<float>(FID_X, *pir, 1);
    rw_xy.write<float>(FID_Y, *pir, 2);
  }
  
  for (RW_Region<1>::iterator pir(rw_xy); pir(); pir++) {
    float x = rw_xy.read<float>(FID_X, *pir);
    float y = rw_xy.read<float>(FID_Y, *pir);
    wd_z.write<float>(FID_Z, *pir, x * alpha + y);
  }
  
  
  wd_z.unmap_physical_inline(c);
  auto ro_z = RO_Region<1>(&output_lr);
  ro_z.map_physical_inline(c);
  
  for (RO_Region<1>::iterator pir(ro_z); pir(); pir++) {
    float x = rw_xy.read<float>(FID_X, *pir);
    float y = rw_xy.read<float>(FID_Y, *pir);
    float z = ro_z.read<float>(FID_Z, *pir);
    printf("x %f, y %f, z %f\n", x, y, z);
  }
  
}

int main(int argc, char** argv){
  runtime.register_task<decltype(&top_level), top_level>("top_level");
  runtime.register_task<decltype(&saxpy), saxpy>("saxpy");
  runtime.register_task<decltype(&check), check>("check");
  runtime.register_task<decltype(&init_value), init_value>("init_value");
  runtime.start(top_level, argc, argv);
}
