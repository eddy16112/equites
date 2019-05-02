#include <legion_simplified.h>
using namespace LegionSimplified;

enum FieldIDs {
  FID_X = 3,
  FID_Y,
};

void top_level(context c)
{ 
  int idx= 0;
  
  IdxSpace<1> ispace(c, 12);
  IdxSpace<1> color_is(c, 4);
  IdxSpace<1> color_is2(c, 2);
  
  FdSpace input_fs(c);
  input_fs.add_field<float>(FID_X);
  input_fs.add_field<float>(FID_Y);
  Region<1> input_lr(ispace, input_fs);
  
  auto rw_xy = RW_Region<1>(input_lr);
  rw_xy.map_physical_region_inline();
  idx = 0;
  for (auto pir : rw_xy) {
    rw_xy.write<float>(FID_X, pir, idx);
    rw_xy.write<float>(FID_Y, pir, idx+1);
    idx++;
  }
  rw_xy.unmap_physical_region_inline();
  printf("!create rw_xy done\n");
  auto ro_xy_par = rw_xy.create_ro_partition(equal, color_is2);
  printf("!create ro_xy_par done\n");
  auto wd_xy_par = rw_xy.create_wd_partition(equal, color_is2);
  printf("!create wd_xy_par done\n");
  auto rw_xy_par = rw_xy.create_rw_partition(equal, color_is2);
  printf("!create rw_xy_par done\n");
  
  Partition<1> input_lp(equal, input_lr, color_is);
  
  std::vector<field_id_t> x_vec{FID_X};
  auto rw_x_par = RW_Partition<1>(input_lp, x_vec);
  printf("!create rw_x_par done\n");
  
  auto ro_x_sub_1 = rw_x_par.get_ro_subregion_by_color(0);
  ro_x_sub_1.map_physical_region_inline();
  for (auto pir : ro_x_sub_1) {
    float x = ro_x_sub_1.read<float>(FID_X, pir);
    printf("ro_x %f\n", x);
  }
  ro_x_sub_1.unmap_physical_region_inline();
  printf("!create ro_x_sub_1 done\n");
  auto wd_x_sub_1 = rw_x_par.get_wd_subregion_by_color(1);
  printf("!create wd_x_sub_1 done\n");
  auto rw_x_sub_1 = rw_x_par.get_rw_subregion_by_color(2);
  rw_x_sub_1.map_physical_region_inline();
  for (auto pir : rw_x_sub_1) {
    float x = rw_x_sub_1.read<float>(FID_X, pir);
    printf("rw_x %f\n", x);
  }
  rw_x_sub_1.unmap_physical_region_inline();
  printf("!create rw_x_sub_1 done\n");
  
}

int main(int argc, char** argv){
  runtime.register_task<decltype(&top_level), top_level>("top_level");
  runtime.start(top_level, argc, argv);
}
