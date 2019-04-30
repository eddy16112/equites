#include <legion_simplified.h>
using namespace LegionSimplified;

enum FieldIDs {
  FID_X = 3,
  FID_Y,
};

void top_level(context c)
{ 
  IdxSpace<1> ispace(c, 120);
  IdxSpace<1> color_is(c, 4);
  IdxSpace<1> color_is2(c, 2);
  
  FdSpace input_fs(c);
  input_fs.add_field<float>(FID_X);
  input_fs.add_field<float>(FID_Y);
  Region<1> input_lr(ispace, input_fs);
  
  Partition<1> input_lp(equal, input_lr, color_is);
  
  std::vector<field_id_t> x_vec{FID_X};
  auto rw_x_par = RW_Partition<1>(input_lp, x_vec);
  printf("!create rw_x_par done\n");
  
  auto ro_x_sub_1 = rw_x_par.get_ro_subregion_by_color(0);
  printf("!create ro_x_sub_1 done\n");

  auto rw_xy = RW_Region<1>(input_lr);
  printf("!create rw_xy done\n");
  auto rw_xy_par = rw_xy.create_ro_partition(equal, color_is2);
  printf("!create rw_xy_par done\n");
  
}

int main(int argc, char** argv){
  runtime.register_task<decltype(&top_level), top_level>("top_level");
  runtime.start(top_level, argc, argv);
}
