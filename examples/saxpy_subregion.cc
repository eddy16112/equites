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
  
  const int point = c.task->index_point.point_data[0];
  /*
  for (RO_Region<1>::iterator pir(region_xy); pir(); pir++) {
    float x = region_xy.read<float>(FID_X, *pir);
    float y = region_xy.read<float>(FID_Y, *pir);
    region_z.write<float>(FID_Z, *pir, x * alpha + y);
  }*/
  printf("saxpy point %d, proc %llx\n", point, c.task->current_proc.id);
}

void init_value(context c, WD_Region<1> region_xy){
  /*
  for (WD_Region<1>::iterator pir(region_xy); pir(); pir++) {
    float value = 2;
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
  
  /*
  for (RW_Region<1>::iterator pir(region_xy); pir(); pir++) {
    float x = region_xy.read<float>(FID_X, *pir);
    float y = region_xy.read<float>(FID_Y, *pir);
    float z = region_z.read<float>(FID_Z, *pir);
    //printf("x %f, y %f, z %f\n", x, y, z);
    if (z != x * alpha + y) {
      printf("error x %f, y %f, z %f\n", x, y, z);
    }
  }
  */
  printf("Success\n");
}

void top_level(context c)
{ 
  IdxSpace<1> ispace(c, 12);
  int num_colors = 4;
  IdxSpace<1> color_is(c, num_colors);
  
  FdSpace input_fs(c);
  input_fs.add_field<float>(FID_X);
  input_fs.add_field<float>(FID_Y);
  Region<1> input_lr(ispace, input_fs);
  
  FdSpace output_fs(c);
  output_fs.add_field<float>(FID_Z);
  Region<1> output_lr(ispace, output_fs);
  
  Partition<1> input_lp(equal, input_lr, color_is);
  Partition<1> output_lp(equal, output_lr, color_is);
  
  std::vector<field_id_t> x_vec{FID_X};
  auto wd_x = WD_Partition<1>(input_lp, x_vec);
  //printf("wd_x1 shared_ptr %p, use_count %ld\n", wd_x.base_region_impl.get(), wd_x.base_region_impl.use_count());
  runtime.execute_task(init_value, c, color_is, wd_x);
  
  std::vector<field_id_t> y_vec{FID_Y};
  auto wd_y = WD_Partition<1>(input_lp, y_vec);
  runtime.execute_task(init_value, c, color_is, wd_y);

  float alpha = 2;
  for (int i = 0; i < num_colors; i++) {
   // Region<1> subregion_xy = input_lp.get_subregion_by_color(i);
    //Region<1> subregion_z = output_lp.get_subregion_by_color(i);
    Region<1> subregion_xy = input_lp[i];
    Region<1> subregion_z = output_lp[i];
    auto ro_subregion_xy = RO_Region<1>(subregion_xy);
    auto wd_subregion_z = WD_Region<1>(subregion_z);
    runtime.execute_task(saxpy, c, alpha, ro_subregion_xy, wd_subregion_z);
  }
  
  auto ro_xy = RO_Partition<1>(input_lp);
  auto wd_z = WD_Partition<1>(output_lp);
  for (int i = 0; i < num_colors; i++) {
   // Region<1> subregion_xy = input_lp.get_subregion_by_color(i);
    //Region<1> subregion_z = output_lp.get_subregion_by_color(i);
    RO_Region<1> ro_subregion_xy = ro_xy.get_ro_subregion_by_color(i);
    WD_Region<1> wd_subregion_z = wd_z.get_wd_subregion_by_color(i);
    runtime.execute_task(saxpy, c, alpha, ro_subregion_xy, wd_subregion_z);
  }

  
  auto ro_xy_all = RO_Region<1>(input_lr);
  auto ro_z_all = RO_Region<1>(output_lr);
  runtime.execute_task(check, c, alpha, ro_xy_all, ro_z_all);
  
  
}

int main(int argc, char** argv){
  runtime.register_task<decltype(&top_level), top_level>("top_level");
  runtime.register_task<decltype(&saxpy), saxpy>("saxpy", true);
  runtime.register_task<decltype(&check), check>("check", true);
  runtime.register_task<decltype(&init_value), init_value>("init_value", true);  
  runtime.start(top_level, argc, argv);
}
