#include <legion_simplified.h>
using namespace LegionSimplified;

enum FieldIDs {
  FID_X = 3,
  FID_Y,
  FID_Z,
};

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
  
  float alpha = 2;
  auto rw_xy = RW_Region<1>(&input_lr);
  auto wd_z = WD_Region<1>(&output_lr);
  
  rw_xy.map_physical_region_inline();
  wd_z.map_physical_region_inline();
  
  for (RW_Region<1>::iterator pir(rw_xy); pir(); pir++) {
    rw_xy.write<float>(FID_X, *pir, 1);
    rw_xy.write<float>(FID_Y, *pir, 2);
  }
  
  for (RW_Region<1>::iterator pir(rw_xy); pir(); pir++) {
    float x = rw_xy.read<float>(FID_X, *pir);
    float y = rw_xy.read<float>(FID_Y, *pir);
    wd_z.write<float>(FID_Z, *pir, x * alpha + y);
  }
  
  //wd_z.unmap_physical_region_inline();
  auto ro_z = RO_Region<1>(&output_lr);
  ro_z.map_physical_region_inline();
  
  for (RO_Region<1>::iterator pir(ro_z); pir(); pir++) {
    float x = rw_xy.read<float>(FID_X, *pir);
    float y = rw_xy.read<float>(FID_Y, *pir);
    float z = ro_z.read<float>(FID_Z, *pir);
    printf("x %f, y %f, z %f\n", x, y, z);
  }
  
  wd_z.map_physical_region_inline();
}

int main(int argc, char** argv){
  runtime.register_task<decltype(&top_level), top_level>("top_level");
  runtime.start(top_level, argc, argv);
}
