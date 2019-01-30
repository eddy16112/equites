#include <legion_simplified.h>
using namespace LegionSimplified;

enum FieldIDs {
  FID_VAL = 3,
  FID_DERIV,
};

void init_field_task(context c, WD_Region<1> region){
  for (WD_Region<1>::iterator pir(region); pir(); pir++) {
    double value = rand() % 4 + 1;;
    region.write<double>(*pir, value);
  }
}

void check_task(context c, int max_elements, RO_Region<1> region){
  bool all_passed = true;
  for (RO_Region<1>::iterator pir(region); pir(); pir++) {
    double l2, l1, r1, r2;
    if (pir[0] < 2)
      l2 = region.read<double>(FID_VAL, 0);
    else
      l2 = region.read<double>(FID_VAL, *pir-2);
    if (pir[0] < 1)
      l1 = region.read<double>(FID_VAL, 0);
    else
      l1 = region.read<double>(FID_VAL, *pir-1);
    if (pir[0] > (max_elements-2))
      r1 = region.read<double>(FID_VAL, max_elements-1);
    else
      r1 = region.read<double>(FID_VAL, *pir+1);
    if (pir[0] > (max_elements-3))
      r2 = region.read<double>(FID_VAL, max_elements-1);
    else
      r2 = region.read<double>(FID_VAL, *pir+2);
    
    double expected = l2+l1+r1+r2;
    double received = region.read<double>(FID_DERIV, *pir);
    if (expected != received)
      all_passed = false;
    
    double val = region.read<double>(FID_VAL, *pir);
    double deriv = region.read<double>(FID_DERIV, *pir);
    printf("val %f, deriv %f\n", val, deriv);
  }
  if (all_passed)
    printf("SUCCESS!\n");
  else
    printf("FAILURE!\n");
}

void stencil_task(context c, int max_elements, RO_Region<1> region_val, RW_Region<1> region_deriv){
  printf("max_elements %d\n", max_elements);
  
  Legion::Domain rect = region_deriv.base_region_impl->domain;
  int lo = rect.lo().get_point<1>();
  int hi = rect.hi().get_point<1>();
  if ((lo < 2) || (hi > (max_elements-3))) {
    for (RW_Region<1>::iterator pir(region_deriv); pir(); pir++) {
      double l2, l1, r1, r2;
      if (pir[0] < 2)
        l2 = region_val.read<double>(FID_VAL, 0);
      else
        l2 = region_val.read<double>(FID_VAL, *pir-2);
      if (pir[0] < 1)
        l1 = region_val.read<double>(FID_VAL, 0);
      else
        l1 = region_val.read<double>(FID_VAL, *pir-1);
      if (pir[0] > (max_elements-2))
        r1 = region_val.read<double>(FID_VAL, max_elements-1);
      else
        r1 = region_val.read<double>(FID_VAL, *pir+1);
      if (pir[0] > (max_elements-3))
        r2 = region_val.read<double>(FID_VAL, max_elements-1);
      else
        r2 = region_val.read<double>(FID_VAL, *pir+2);
      
      double result = l2+l1+r1+r2;
      region_deriv.write<double>(FID_DERIV, *pir, result);
    }
  } else {
    for (RW_Region<1>::iterator pir(region_deriv); pir(); pir++) {
      double l2 = region_val.read<double>(FID_VAL, *pir-2);
      double l1 = region_val.read<double>(FID_VAL, *pir-1);
      double r1 = region_val.read<double>(FID_VAL, *pir+1);
      double r2 = region_val.read<double>(FID_VAL, *pir+2);
      double result = l2+l1+r1+r2;
      region_deriv.write<double>(FID_DERIV, *pir, result);
    }
  }
}

void top_level(context c)
{ 
  int num_elements = 160;
  int num_subregions = 8;
  IdxSpace<1> ispace(c, num_elements);
  IdxSpace<1> color_is(c, num_subregions);
  
  FdSpace fs(c);
  fs.add_field<double>(FID_VAL);
  fs.add_field<double>(FID_DERIV);
  Region<1> stencil_lr(ispace, fs);
  
  Partition<1> disjoint_lp(equal, stencil_lr, color_is);
  
  const int block_size = (num_elements + num_subregions - 1) / num_subregions;
  
  Legion::Transform<1,1> transform;
  transform[0][0] = block_size;
  Rect<1> extent(-2, block_size + 1);
  
  Legion::DomainTransform d_transform(transform);
  Partition<1> ghost_lp(restriction, stencil_lr, color_is, d_transform, extent);
  
  std::vector<field_id_t> val_vec{FID_VAL};
  auto wd_val = WD_Region<1>(disjoint_lp, val_vec);
  runtime.execute_task(init_field_task, c, color_is, wd_val);
  
  std::vector<field_id_t> deriv_vec{FID_DERIV};
  auto ro_val = RO_Region<1>(ghost_lp, val_vec);
  auto rw_deriv = RW_Region<1>(disjoint_lp, deriv_vec);
  runtime.execute_task(stencil_task, c, color_is, num_elements, ro_val, rw_deriv);
  
  auto ro_all = RO_Region<1>(stencil_lr);
  runtime.execute_task(check_task, c, num_elements, ro_all);
  
}

int main(int argc, char** argv){
  runtime.register_task<decltype(&top_level), top_level>("top_level");
  runtime.register_task<decltype(&stencil_task), stencil_task>("stencil_task", true);
  runtime.register_task<decltype(&check_task), check_task>("check_task", true);
  runtime.register_task<decltype(&init_field_task), init_field_task>("init_field_task", true);
  runtime.start(top_level, argc, argv);
}
