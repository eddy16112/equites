#include "legion_simplified.h"

namespace LegionSimplified {

  /////////////////////////////////////////////////////////////
  // IdxSpace
  /////////////////////////////////////////////////////////////
  
  //----------------------------------public-------------------------------------
  template <size_t DIM>
  IdxSpace<DIM>::IdxSpace(const context& c, Point<DIM> p) : ctx(c)
  {
    Rect<DIM> rect = Rect<DIM>(Point<DIM>::ZEROES(), p-Point<DIM>::ONES());
    std::cout << "ispace set rect to be from " << Point<DIM>::ZEROES() 
      << " to " << rect.hi << std::endl; 
    index_space = c.runtime->create_index_space(c.ctx, rect); 
  }
  
  template <size_t DIM>
  IdxSpace<DIM>::~IdxSpace(void)
  {
    DEBUG_PRINT((4, "IdxSpace destructor %p\n", this));
    ctx.runtime->destroy_index_space(ctx.ctx, index_space);
  }
  
  /////////////////////////////////////////////////////////////
  // FdSpace
  /////////////////////////////////////////////////////////////
  
  //----------------------------------public-------------------------------------
  template <typename T>
  void FdSpace::add_field(field_id_t field_id)
  {
    allocator.allocate_field(sizeof(T),field_id);
    field_id_vector.push_back(field_id);
  }
  
  /////////////////////////////////////////////////////////////
  // Region
  /////////////////////////////////////////////////////////////
  
  //----------------------------------public-------------------------------------
  template <size_t DIM>
  Region<DIM>::Region(IdxSpace<DIM> &ispace, FdSpace &fspace) : 
    ctx(fspace.ctx), field_id_vector(fspace.field_id_vector)
  {
    DEBUG_PRINT((4, "Region constructor %p\n", this));
    logical_region = ctx.runtime->create_logical_region(ctx.ctx, ispace.index_space, fspace.field_space);
    logical_region_parent = logical_region;
  }
  
  template <size_t DIM>
  Region<DIM>::Region(const context &c, const std::vector<field_id_t> &field_id_vec, Legion::LogicalRegion &lr, Legion::LogicalRegion &lr_parent) :
    ctx(c), field_id_vector(field_id_vec), logical_region(lr), logical_region_parent(lr_parent)
  {
    DEBUG_PRINT((4, "Region constructor all %p\n", this));
  }
  
  template <size_t DIM>
  Region<DIM>::~Region(void)
  {
    DEBUG_PRINT((4, "Region destructor %p\n", this));
    // if I am the parent logical region, then destroy it.
    if (logical_region == logical_region_parent) {
      DEBUG_PRINT((4, "Region destructor destroy lr %p\n", this));
      ctx.runtime->destroy_logical_region(ctx.ctx, logical_region);
    }
  }
  
  /////////////////////////////////////////////////////////////
  // Partition
  /////////////////////////////////////////////////////////////
  
  //----------------------------------public-------------------------------------
  template <size_t DIM>
  Partition<DIM>::Partition(enum partition_type p_type, Region<DIM> &r, IdxSpace<DIM> &ispace) : 
    ctx(r.ctx), field_id_vector(r.field_id_vector), logical_region_parent(r.logical_region_parent)
  {
    index_partition = ctx.runtime->create_equal_partition(ctx.ctx, r.logical_region.get_index_space(), ispace.index_space);
    logical_partition = ctx.runtime->get_logical_partition(ctx.ctx, r.logical_region, index_partition);
  }
  
  template <size_t DIM>
  Partition<DIM>::Partition(enum partition_type p_type, Region<DIM> &r, IdxSpace<DIM> &ispace, 
    Legion::DomainTransform &dt, Rect<DIM> &rect) : 
    ctx(r.ctx), field_id_vector(r.field_id_vector), logical_region_parent(r.logical_region_parent)
  {
    index_partition = ctx.runtime->create_partition_by_restriction(ctx.ctx, r.logical_region.get_index_space(), ispace.index_space, dt, rect);
    logical_partition = ctx.runtime->get_logical_partition(ctx.ctx, r.logical_region, index_partition);
  }
  
  template <size_t DIM>
  Partition<DIM>::~Partition(void)
  {
  }
  
  
  template <size_t DIM>
  Region<DIM> Partition<DIM>::get_subregion_by_color(int color)
  {
    Legion::LogicalRegion sub_lr = ctx.runtime->get_logical_subregion_by_color(ctx.ctx, logical_partition, color);
    Region<DIM> subregion(ctx, field_id_vector, sub_lr, logical_region_parent);
    return subregion; 
  }
  
  template <size_t DIM>
  Region<DIM> Partition<DIM>::operator[](int color)
  {
    return get_subregion_by_color(color); 
  }
  
  /////////////////////////////////////////////////////////////
  // Base_Region 
  /////////////////////////////////////////////////////////////
  
  //----------------------------------public-------------------------------------
  template <size_t DIM>
  Base_Region<DIM>::Base_Region(void)
  {
    init_parameters();
    DEBUG_PRINT((4, "Base_Region empty constructor %p\n", this));
  } 

  template <size_t DIM>
  Base_Region<DIM>::Base_Region(const Base_Region &rhs)
  {
    DEBUG_PRINT((8, "Base_Region copy constructor %p\n", this));
    init_parameters();
    this->base_region_impl = rhs.base_region_impl;
    this->pm = rhs.pm;
  }

  template <size_t DIM>
  Base_Region<DIM>::Base_Region(Region<DIM> &r, std::vector<field_id_t> &task_field_id_vec) 
  {
    DEBUG_PRINT((4, "Base_Region Region/field constructor %p\n", this));
    init_parameters();
    base_region_impl = std::make_shared<BaseRegionImpl>(r.ctx);
    base_region_impl->logical_region = r.logical_region;
    base_region_impl->logical_region_parent = r.logical_region_parent;
    std::vector<field_id_t>::const_iterator it; 
    for (it = task_field_id_vec.cbegin(); it != task_field_id_vec.cend(); it++) {
       printf("base set fid %d\n", *it);
       base_region_impl->field_id_vector.push_back(*it); 
    }
    base_region_impl->domain = base_region_impl->ctx.runtime->get_index_space_domain(base_region_impl->ctx.ctx, base_region_impl->logical_region.get_index_space());
  }

  template <size_t DIM>
  Base_Region<DIM>::Base_Region(Region<DIM> &r) 
  {
    DEBUG_PRINT((4, "Base_Region Region constructor %p\n", this));
    init_parameters();
    base_region_impl = std::make_shared<BaseRegionImpl>(r.ctx);
    base_region_impl->logical_region = r.logical_region;
    base_region_impl->logical_region_parent = r.logical_region_parent;
    const std::vector<field_id_t> &task_field_id_vec = r.field_id_vector;
    std::vector<field_id_t>::const_iterator it; 
    for (it = task_field_id_vec.cbegin(); it != task_field_id_vec.cend(); it++) {
       DEBUG_PRINT((6, "Base_Region %p, set fid %d\n", this, *it));
       base_region_impl->field_id_vector.push_back(*it); 
    }
    base_region_impl->domain = base_region_impl->ctx.runtime->get_index_space_domain(base_region_impl->ctx.ctx, base_region_impl->logical_region.get_index_space());
  }

  template <size_t DIM>
  Base_Region<DIM>::Base_Region(Partition<DIM> &par, std::vector<field_id_t> &task_field_id_vec)
  {
    DEBUG_PRINT((4, "Base_Region Partition/field constructor %p\n", this));
    init_parameters();
    base_region_impl = std::make_shared<BaseRegionImpl>(par.ctx);
    base_region_impl->logical_partition = par.logical_partition;
    base_region_impl->logical_region_parent = par.logical_region_parent;
    std::vector<field_id_t>::const_iterator it; 
    for (it = task_field_id_vec.cbegin(); it != task_field_id_vec.cend(); it++) {
       DEBUG_PRINT((6, "Base_Region %p, set fid %d\n", this, *it));
       base_region_impl->field_id_vector.push_back(*it); 
    }
    //base_region_impl->domain = ctx->runtime->get_index_space_domain(ctx->ctx, base_region_impl->lr.get_index_space());
  }

  template <size_t DIM>
  Base_Region<DIM>::Base_Region(Partition<DIM> &par)
  {
    DEBUG_PRINT((4, "Base_Region Partition constructor %p\n", this));
    init_parameters();
    base_region_impl = std::make_shared<BaseRegionImpl>(par.ctx);
    base_region_impl->logical_partition = par.logical_partition;
    base_region_impl->logical_region_parent = par.logical_region_parent;
    const std::vector<field_id_t> &task_field_id_vec = par.field_id_vector;
    std::vector<field_id_t>::const_iterator it; 
    for (it = task_field_id_vec.cbegin(); it != task_field_id_vec.cend(); it++) {
       DEBUG_PRINT((6, "Base_Region %p, set fid %d\n", this, *it));
       base_region_impl->field_id_vector.push_back(*it); 
    }
   // base_region_impl->domain = ctx->runtime->get_index_space_domain(ctx->ctx, base_region_impl->lr.get_index_space());
  }
  
  template <size_t DIM>
  Base_Region<DIM>::~Base_Region(void)
  {
    /*
    if (base_region_impl != nullptr) {
      if (base_region_impl->is_mapped == PR_INLINE_MAPPED) {
        unmap_physical_region_inline();
      }
    }
    */
    /*
    std::map<field_id_t, unsigned char*>::iterator it; 
    for (it = accessor_map.begin(); it != accessor_map.end(); it++) {
      if (it->second != NULL) {
        printf("free accessor of fid %d\n", it->first);
        delete it->second;
        it->second = NULL;
      }
    }*/

 /*   
    if (logical_region_impl != NULL) {
      printf("free %p\n", logical_region_impl);
      delete logical_region_impl;
      logical_region_impl = NULL;
    }*/
    /*
    if (base_region_impl != NULL) {
      printf("THIS %p, impl %p, remove reference free%d\n", this, base_region_impl, base_region_impl->references);
      if (base_region_impl->remove_reference()) {
        printf("$$$$$$$$$$free THIS %p, %p\n", this, base_region_impl);
        delete base_region_impl;
      }
      base_region_impl = NULL;
    }*/
    long use_count = 0;
    void *shared_ptr = nullptr;
    if (base_region_impl != nullptr) 
    {
      use_count = base_region_impl.use_count();
      shared_ptr = base_region_impl.get();
    } else {
      use_count = -1;
    }
    DEBUG_PRINT((4, "Base_Region %p destructor, shared_ptr %p, use count %ld\n", this, shared_ptr, use_count));
  }

  template <size_t DIM>
  Base_Region<DIM> & Base_Region<DIM>::operator=(const Base_Region &rhs)
  {
    init_parameters();
    this->ctx = rhs.ctx;
    this->base_region_impl = rhs.base_region_impl;
    this->pm = rhs.pm;
  }
  
  template <size_t DIM>
  Legion::RegionRequirement Base_Region<DIM>::set_region_requirement_single(void)
  {
    Legion::RegionRequirement req(base_region_impl->logical_region, pm, cp, base_region_impl->logical_region_parent);
    std::vector<field_id_t>::iterator it; 
    for (it = base_region_impl->field_id_vector.begin(); it < base_region_impl->field_id_vector.end(); it++) {
      DEBUG_PRINT((4, "Base_Region %p, set RR fid %d for single launcher\n", this, *it));
      req.add_field(*it); 
    }
    return req; 
  }

  template <size_t DIM>
  Legion::RegionRequirement Base_Region<DIM>::set_region_requirement_index(void)
  {
    Legion::RegionRequirement req(base_region_impl->logical_partition, 0, pm, cp, base_region_impl->logical_region_parent);
    std::vector<field_id_t>::iterator it; 
    for (it = base_region_impl->field_id_vector.begin(); it < base_region_impl->field_id_vector.end(); it++) {
      DEBUG_PRINT((4, "Base_Region %p, set RR fid %d for index launcher\n", this, *it));
      req.add_field(*it); 
    }
    return req; 
  }

  template <size_t DIM>
  void Base_Region<DIM>::map_physical_region(context &c, Legion::PhysicalRegion &pr, Legion::RegionRequirement &rr)
  {
    check_empty();
    init_parameters();
    base_region_impl = std::make_shared<BaseRegionImpl>(c);
    DEBUG_PRINT((4, "Base_Region %p, map_physical_region, BaseRegionImpl shared_ptr %p\n", this, base_region_impl.get()));
  
    base_region_impl->logical_region = pr.get_logical_region();
    base_region_impl->logical_region_parent = rr.parent;
    base_region_impl->physical_region = pr;
    std::set<field_id_t>::iterator it;
    for (it = rr.privilege_fields.begin(); it != rr.privilege_fields.end(); it++) {
      base_region_impl->field_id_vector.push_back(*it);
      DEBUG_PRINT((4, "Base_Region %p, map_physical_region rr field %d, set accessor \n", this, *it));
      unsigned char *null_ptr = nullptr;
      base_region_impl->accessor_map.insert(std::make_pair(*it, null_ptr)); 
    }
    base_region_impl->domain = c.runtime->get_index_space_domain(c.ctx, rr.region.get_index_space());
    base_region_impl->is_mapped = PR_TASK_MAPPED;
  }
  
  template <size_t DIM>
  void Base_Region<DIM>::map_physical_region_inline_with_auto_unmap()
  {
  }
  
  template <size_t DIM>
  void Base_Region<DIM>::map_physical_region_inline(void)
  {
    if (base_region_impl->is_mapped != PR_NOT_MAPPED) {
      return;
    }
    DEBUG_PRINT((4, "Base_Region %p, map_physical_region_inline, BaseRegionImpl shared_ptr %p\n", this, base_region_impl.get()));
    Legion::RegionRequirement req(base_region_impl->logical_region, pm, cp, base_region_impl->logical_region_parent);
    std::vector<field_id_t>::iterator it; 
    for (it = base_region_impl->field_id_vector.begin(); it < base_region_impl->field_id_vector.end(); it++) {
      DEBUG_PRINT((4, "Base_Region %p, inline map for fid %d\n", this, *it));
      req.add_field(*it);
      unsigned char *null_ptr = NULL;
      base_region_impl->accessor_map.insert(std::make_pair(*it, null_ptr));  
    }
    base_region_impl->physical_region = base_region_impl->ctx.runtime->map_region(base_region_impl->ctx.ctx, req);
    base_region_impl->physical_region.wait_until_valid();
    base_region_impl->domain = base_region_impl->ctx.runtime->get_index_space_domain(base_region_impl->ctx.ctx, req.region.get_index_space());
    base_region_impl->is_mapped = PR_INLINE_MAPPED;
  }
  
  template <size_t DIM>
  void Base_Region<DIM>::unmap_physical_region_inline(void)
  {
    if (base_region_impl->is_mapped == PR_INLINE_MAPPED) {
      base_region_impl->unmap_physical_region();
      DEBUG_PRINT((4, "Base_Region %p, unmap region inline\n", this));
    }
  }
  
  template <size_t DIM>
  void Base_Region<DIM>::cleanup_reference(void)
  {
    if (base_region_impl != nullptr) {
    //  auto tmp = base_region_impl;
      DEBUG_PRINT((4, "Base_Region %p, reset base_region_impl %p, count %ld\n", this, base_region_impl.get(), base_region_impl.use_count()));
      base_region_impl.reset();
  //    DEBUG_PRINT((4, "Base_Region %p, reset tmp %p, count %ld\n", this, tmp.get(), tmp.use_count()));
      base_region_impl = nullptr;
    //  DEBUG_PRINT((4, "after nullptr %ld\n", tmp.use_count()));
    }
  }
  
  template <size_t DIM>
  void Base_Region<DIM>::if_mapped(void)
  { 
    // not mapped, do inline mapping
    if (base_region_impl->is_mapped == PR_NOT_MAPPED) {
      map_physical_region_inline_with_auto_unmap();
    }
  }
  
  template <size_t DIM>
  Region<DIM> Base_Region<DIM>::get_region(void)
  {
    // fixme 
    Region<DIM> region = Region<DIM>(base_region_impl->ctx, base_region_impl->field_id_vector, base_region_impl->logical_region, base_region_impl->logical_region_parent);
    return region;
  }
  
  //----------------------------------private-------------------------------------
  template <size_t DIM>
  void Base_Region<DIM>::init_parameters(void)
  {
    base_region_impl = nullptr;
  }

  template <size_t DIM>
  void Base_Region<DIM>::check_empty(void)
  {
    assert(base_region_impl == nullptr);
    
  } 
  
  /////////////////////////////////////////////////////////////
  // RO_Region
  /////////////////////////////////////////////////////////////
  
  //----------------------------------public-------------------------------------
  template <size_t DIM>
  RO_Region<DIM>::RO_Region(void) : Base_Region<DIM>()
  {
  }
  
  template <size_t DIM>
  RO_Region<DIM>::RO_Region(Region<DIM> &r, std::vector<field_id_t> &task_field_id_vec) 
    : Base_Region<DIM>(r, task_field_id_vec)
  {
    init_ro_parameters();
  }

  template <size_t DIM>
  RO_Region<DIM>::RO_Region(Region<DIM> &r) 
    : Base_Region<DIM>(r)
  {
    init_ro_parameters();
  }

  template <size_t DIM>
  RO_Region<DIM>::RO_Region(Partition<DIM> &par, std::vector<field_id_t> &task_field_id_vec) 
    : Base_Region<DIM>(par, task_field_id_vec)
  {
    init_ro_parameters();
  }

  template <size_t DIM>
  RO_Region<DIM>::RO_Region(Partition<DIM> &par) 
    : Base_Region<DIM>(par)
  {
    init_ro_parameters();
  }

  template <size_t DIM>
  RO_Region<DIM>::~RO_Region(void)
  {
  }
  
  template <size_t DIM>
  template <typename a>
  a RO_Region<DIM>::read(int fid, Legion::Point<DIM> i)
  {
    Base_Region<DIM>::if_mapped();
    assert(this->base_region_impl->is_mapped != PR_NOT_MAPPED);
    Legion::FieldAccessor<READ_ONLY, a, DIM> *acc = get_accessor_by_fid<a>(fid);
    return (*acc)[i];
  }
  
  template <size_t DIM>
  template <typename a>
  a RO_Region<DIM>::read(Legion::Point<DIM> i)
  {
    Base_Region<DIM>::if_mapped();
    assert(this->base_region_impl->is_mapped != PR_NOT_MAPPED);
    assert(this->base_region_impl->accessor_map.size() == 1);
    Legion::FieldAccessor<READ_ONLY, a, DIM> *acc = get_default_accessor<a>();
    return (*acc)[i];
  }
  
  //----------------------------------protected-------------------------------------
  template <size_t DIM>
  void RO_Region<DIM>::init_ro_parameters(void)
  {
    this->pm = READ_ONLY;
  }
  
  //----------------------------------private-------------------------------------
  
  template <size_t DIM>
  template< typename a>
  Legion::FieldAccessor<READ_ONLY, a, DIM>* RO_Region<DIM>::get_accessor_by_fid(field_id_t fid)
  {
    typename std::map<field_id_t, unsigned char*>::iterator it = this->base_region_impl->accessor_map.find(fid);
    if (it != this->base_region_impl->accessor_map.end()) {
      if (it->second == nullptr) {
        DEBUG_PRINT((2, "RO_Region %p, first time create accessor for fid %d\n", this, fid));
        Legion::FieldAccessor<READ_ONLY, a, DIM> *acc = new Legion::FieldAccessor<READ_ONLY, a, DIM>(this->base_region_impl->physical_region, fid);
        it->second = (unsigned char*)acc;
      }
      return (Legion::FieldAccessor<READ_ONLY, a, DIM>*)(it->second);
    } else {
      DEBUG_PRINT((0, "RO_Region %p, can not find accessor of fid %d\n", this, fid));
      assert(0);
      return nullptr;
    }
  }

  template <size_t DIM>
  template< typename a>
  Legion::FieldAccessor<READ_ONLY, a, DIM>* RO_Region<DIM>::get_default_accessor(void)
  {
    assert(this->base_region_impl->field_id_vector.size() == 1);
    return get_accessor_by_fid<a>(this->base_region_impl->field_id_vector[0]);
  }
  
  /////////////////////////////////////////////////////////////
  // WD_Region
  /////////////////////////////////////////////////////////////
  
  //----------------------------------public-------------------------------------
  template <size_t DIM>
  WD_Region<DIM>::WD_Region(void) : Base_Region<DIM>()
  {
  }

  template <size_t DIM>
  WD_Region<DIM>::WD_Region(Region<DIM> &r, std::vector<field_id_t> &task_field_id_vec) 
    : Base_Region<DIM>(r, task_field_id_vec)
  {
    init_wd_parameters();
  }

  template <size_t DIM>
  WD_Region<DIM>::WD_Region(Region<DIM> &r) : Base_Region<DIM>(r)
  {
    init_wd_parameters();
  }

  template <size_t DIM>
  WD_Region<DIM>::WD_Region(Partition<DIM> &par, std::vector<field_id_t> &task_field_id_vec) 
    : Base_Region<DIM>(par, task_field_id_vec)
  {
    init_wd_parameters();
  }

  template <size_t DIM>
  WD_Region<DIM>::WD_Region(Partition<DIM> &par) 
    : Base_Region<DIM>(par)
  {
    init_wd_parameters();
  }

  template <size_t DIM>
  WD_Region<DIM>::~WD_Region(void)
  {
  }
  
  template <size_t DIM>
  template< typename a>
  void WD_Region<DIM>::write(int fid, Legion::Point<DIM> i, a x)
  {
    Base_Region<DIM>::if_mapped();
    assert(this->base_region_impl->is_mapped != PR_NOT_MAPPED);
    Legion::FieldAccessor<WRITE_DISCARD, a, DIM> *acc = get_accessor_by_fid<a>(fid);
    (*acc)[i] = x; 
  }

  template <size_t DIM>
  template< typename a>
  void WD_Region<DIM>::write(Legion::Point<DIM> i, a x)
  {
    Base_Region<DIM>::if_mapped();
    assert(this->base_region_impl->is_mapped != PR_NOT_MAPPED);
    assert(this->base_region_impl->accessor_map.size() == 1);
    Legion::FieldAccessor<WRITE_DISCARD, a, DIM> *acc = get_default_accessor<a>();
    (*acc)[i] = x; 
  }
  
  //----------------------------------protected-------------------------------------
  template <size_t DIM>
  void WD_Region<DIM>::init_wd_parameters(void)
  {
    this->pm = WRITE_DISCARD;
  }
  
  //----------------------------------private-------------------------------------

  template <size_t DIM>
  template< typename a>
  Legion::FieldAccessor<WRITE_DISCARD, a, DIM>* WD_Region<DIM>::get_accessor_by_fid(field_id_t fid)
  {
    typename std::map<field_id_t, unsigned char*>::iterator it = this->base_region_impl->accessor_map.find(fid);
    if (it != this->base_region_impl->accessor_map.end()) {
      if (it->second == nullptr) {
        DEBUG_PRINT((2, "WD_Region %p, first time create accessor for fid %d\n", this, fid));
        Legion::FieldAccessor<WRITE_DISCARD, a, DIM> *acc = new Legion::FieldAccessor<WRITE_DISCARD, a, DIM>(this->base_region_impl->physical_region, fid);
        it->second = (unsigned char*)acc;
      }
      return (Legion::FieldAccessor<WRITE_DISCARD, a, DIM>*)(it->second);
    } else {
      DEBUG_PRINT((0, "WD_Region %p, can not find accessor of fid %d\n", this, fid));
      assert(0);
      return nullptr;
    }
  }

  template <size_t DIM>
  template< typename a>
  Legion::FieldAccessor<WRITE_DISCARD, a, DIM>* WD_Region<DIM>::get_default_accessor(void)
  {
    assert(this->base_region_impl->field_id_vector.size() == 1);
    return get_accessor_by_fid<a>(this->base_region_impl->field_id_vector[0]);
  } 
  
  /////////////////////////////////////////////////////////////
  // RW_Region
  /////////////////////////////////////////////////////////////
  
  //----------------------------------public-------------------------------------
  template <size_t DIM>
  RW_Region<DIM>::RW_Region(void) 
    : Base_Region<DIM>()
  {
  }

  template <size_t DIM>
  RW_Region<DIM>::RW_Region(Region<DIM> &r, std::vector<field_id_t> &task_field_id_vec) 
    : Base_Region<DIM>(r, task_field_id_vec)
  {
    init_rw_parameters();
  }

  template <size_t DIM>
  RW_Region<DIM>::RW_Region(Region<DIM> &r) 
    : Base_Region<DIM>(r)
  {
    init_rw_parameters();
  }

  template <size_t DIM>
  RW_Region<DIM>::RW_Region(Partition<DIM> &par, std::vector<field_id_t> &task_field_id_vec) 
    : Base_Region<DIM>(par, task_field_id_vec)
  {
    init_rw_parameters();
  }

  template <size_t DIM>
  RW_Region<DIM>::RW_Region(Partition<DIM> &par) 
    : Base_Region<DIM>(par)
  {
    init_rw_parameters();
  }

  template <size_t DIM>
  RW_Region<DIM>::~RW_Region(void)
  {
  }
  
  template <size_t DIM>
  template< typename a>
  a RW_Region<DIM>::read(int fid, Legion::Point<DIM> i)
  {
    Base_Region<DIM>::if_mapped();
    assert(this->base_region_impl->is_mapped != PR_NOT_MAPPED);
    Legion::FieldAccessor<READ_WRITE, a, DIM> *acc = get_accessor_by_fid<a>(fid);
    return (*acc)[i];
  };
  
  template <size_t DIM>
  template< typename a>
  a RW_Region<DIM>::read(Legion::Point<DIM> i)
  {
    Base_Region<DIM>::if_mapped();
    assert(this->base_region_impl->is_mapped != PR_NOT_MAPPED);
    assert(this->base_region_impl->accessor_map.size() == 1);
    Legion::FieldAccessor<READ_WRITE, a, DIM> *acc = get_default_accessor<a>();
    return (*acc)[i];
  };

  template <size_t DIM>
  template< typename a>
  void RW_Region<DIM>::write(int fid, Legion::Point<DIM> i, a x)
  {
    Base_Region<DIM>::if_mapped();
    assert(this->base_region_impl->is_mapped != PR_NOT_MAPPED);
    Legion::FieldAccessor<READ_WRITE, a, DIM> *acc = get_accessor_by_fid<a>(fid);
    (*acc)[i] = x; 
  }

  template <size_t DIM>
  template< typename a>
  void RW_Region<DIM>::write(Legion::Point<DIM> i, a x)
  {
    Base_Region<DIM>::if_mapped();
    assert(this->base_region_impl->is_mapped != PR_NOT_MAPPED);
    assert(this->base_region_impl->accessor_map.size() == 1);
    Legion::FieldAccessor<READ_WRITE, a, DIM> *acc = get_default_accessor<a>();
    (*acc)[i] = x; 
  }
  
  //----------------------------------protected-------------------------------------
  template <size_t DIM>
  void RW_Region<DIM>::init_rw_parameters(void)
  {
    this->pm = READ_WRITE;
  }
  
  //----------------------------------private-------------------------------------

  template <size_t DIM>
  template< typename a>
  Legion::FieldAccessor<READ_WRITE, a, DIM>* RW_Region<DIM>::get_accessor_by_fid(field_id_t fid)
  {
    typename std::map<field_id_t, unsigned char*>::iterator it = this->base_region_impl->accessor_map.find(fid);
    if (it != this->base_region_impl->accessor_map.end()) {
      if (it->second == nullptr) {
        DEBUG_PRINT((2, "RW_Region %p, first time create accessor for fid %d\n", this, fid));
        Legion::FieldAccessor<READ_WRITE, a, DIM> *acc = new Legion::FieldAccessor<READ_WRITE, a, DIM>(this->base_region_impl->physical_region, fid);
        it->second = (unsigned char*)acc;
      }
      return (Legion::FieldAccessor<READ_WRITE, a, DIM>*)(it->second);
    } else {
      DEBUG_PRINT((0, "RW_Region %p, can not find accessor of fid %d\n", this, fid));
      assert(0);
      return nullptr;
    }
  }

  template <size_t DIM>
  template< typename a>
  Legion::FieldAccessor<READ_WRITE, a, DIM>* RW_Region<DIM>::get_default_accessor(void)
  {
    assert(this->base_region_impl->field_id_vector.size() == 1);
    return get_accessor_by_fid<a>(this->base_region_impl->field_id_vector[0]);
  }
  
  /////////////////////////////////////////////////////////////
  // RO_Partition
  /////////////////////////////////////////////////////////////
  
  //----------------------------------public-------------------------------------
  template <size_t DIM>
  RO_Partition<DIM>::RO_Partition(void) : RO_Region<DIM>()
  {
  }

  template <size_t DIM>
  RO_Partition<DIM>::RO_Partition(Partition<DIM> &par, std::vector<field_id_t> &task_field_id_vec) 
    : RO_Region<DIM>(par, task_field_id_vec)
  {
    RO_Region<DIM>::init_ro_parameters();
  }

  template <size_t DIM>
  RO_Partition<DIM>::RO_Partition(Partition<DIM> &par) 
    : RO_Region<DIM>(par)
  {
    RO_Region<DIM>::init_ro_parameters();
  }

  template <size_t DIM>
  RO_Partition<DIM>::~RO_Partition(void)
  {
  }
  
  /////////////////////////////////////////////////////////////
  // WD_Partition
  /////////////////////////////////////////////////////////////
  
  //----------------------------------public-------------------------------------
  template <size_t DIM>
  WD_Partition<DIM>::WD_Partition(void) : WD_Region<DIM>()
  {
  }

  template <size_t DIM>
  WD_Partition<DIM>::WD_Partition(Partition<DIM> &par, std::vector<field_id_t> &task_field_id_vec) 
    : WD_Region<DIM>(par, task_field_id_vec)
  {
    WD_Region<DIM>::init_wd_parameters();
  }

  template <size_t DIM>
  WD_Partition<DIM>::WD_Partition(Partition<DIM> &par) 
    : WD_Region<DIM>(par)
  {
    WD_Region<DIM>::init_wd_parameters();
  }

  template <size_t DIM>
  WD_Partition<DIM>::~WD_Partition(void)
  {
  }
  
  /////////////////////////////////////////////////////////////
  // RW_Partition
  /////////////////////////////////////////////////////////////
  
  //----------------------------------public-------------------------------------
  template <size_t DIM>
  RW_Partition<DIM>::RW_Partition(void) : RW_Region<DIM>()
  {
  }

  template <size_t DIM>
  RW_Partition<DIM>::RW_Partition(Partition<DIM> &par, std::vector<field_id_t> &task_field_id_vec) 
    : RW_Region<DIM>(par, task_field_id_vec)
  {
    RW_Region<DIM>::init_rw_parameters();
  }

  template <size_t DIM>
  RW_Partition<DIM>::RW_Partition(Partition<DIM> &par) 
    : RW_Region<DIM>(par)
  {
    RW_Region<DIM>::init_rw_parameters();
  }

  template <size_t DIM>
  RW_Partition<DIM>::~RW_Partition(void)
  {
  }
  
  /////////////////////////////////////////////////////////////
  // UserTask 
  /////////////////////////////////////////////////////////////
  
  //----------------------------------public-------------------------------------
  template <typename F, F f>
  void UserTask::register_task(bool leaf)
  {
    typedef typename function_traits<F>::returnType RT; 
    Legion::ProcessorConstraint pc = Legion::ProcessorConstraint(Legion::Processor::LOC_PROC);
    Legion::TaskVariantRegistrar registrar(id, task_name.c_str());
    registrar.add_constraint(pc);
    if (leaf == true) {
      registrar.set_leaf();
    }
    TaskRegistration<RT, F, f>::variant(registrar, task_name.c_str()); 
  }
  
  template <typename F, typename ...Args>
  Future UserTask::launch_single_task(F f, context &c, Args... a)
  {
    typedef typename function_traits<F>::args argtuple;
    argtuple p = std::make_tuple(a...);
    //auto r = std::get<0>(p);
    //printf("p1 count %ld\n", r.base_region_impl.use_count());
    argtuple p2 = std::make_tuple(a...);
    //auto r2 = std::get<0>(p2);
    //printf("p2 count %ld\n", r2.base_region_impl.use_count());
    base_region_cleanup_shared_ptr_tuple_walker(p2);
    Legion::TaskLauncher task_launcher(id, Legion::TaskArgument(&p2, sizeof(p2))); 
    task_launcher_region_requirement_tuple_walker(task_launcher, p);  
    return Future(c.runtime->execute_task(c.ctx, task_launcher));
  }
  
  // launch index task  
  template <size_t DIM, typename F, typename ...Args>
  FutureMap UserTask::launch_index_task(F f, context &c, IdxSpace<DIM> &ispace, Args... a){
    typedef typename function_traits<F>::args argtuple;
    argtuple p = std::make_tuple(a...);
    argtuple p2 = std::make_tuple(a...);
    base_region_cleanup_shared_ptr_tuple_walker(p2);
    Legion::ArgumentMap arg_map; 
    Legion::IndexLauncher index_launcher(id, ispace.index_space, Legion::TaskArgument(&p2, sizeof(p2)), arg_map); 
    index_launcher_region_requirement_tuple_walker(index_launcher, p);  
    return FutureMap(c.runtime->execute_index_space(c.ctx, index_launcher));
  }

  // launch index task with argmap
  template <size_t DIM, typename F, typename ...Args>
  FutureMap UserTask::launch_index_task(F f, context &c, IdxSpace<DIM> &ispace, ArgMap argmap, Args... a){
    typedef typename function_traits<F>::args argtuple;
    argtuple p = std::make_tuple(a...);
    argtuple p2 = std::make_tuple(a...);
    base_region_cleanup_shared_ptr_tuple_walker(p2);
    Legion::IndexLauncher index_launcher(id, ispace.index_space, Legion::TaskArgument(&p2, sizeof(p2)), argmap.arg_map); 
    index_launcher_region_requirement_tuple_walker(index_launcher, p);  
    return FutureMap(c.runtime->execute_index_space(c.ctx, index_launcher));
  }
  
  /////////////////////////////////////////////////////////////
  // TaskRuntime 
  /////////////////////////////////////////////////////////////
  
  //----------------------------------public-------------------------------------
  template <typename F, F func_ptr>
  void TaskRuntime::register_task(const char* name)
  {
    UserTask new_task(name);
    new_task.register_task<F, func_ptr>(false);
    user_task_map.insert(std::make_pair((uintptr_t)func_ptr, new_task)); 
  }
  
  template <typename F, F func_ptr>
  void TaskRuntime::register_task(const char* name, bool leaf)
  {
    UserTask new_task(name);
    new_task.register_task<F, func_ptr>(leaf);
    user_task_map.insert(std::make_pair((uintptr_t)func_ptr, new_task)); 
  }

  template <typename F>
  int TaskRuntime::start(F func_ptr, int argc, char** argv)
  { 
    UserTask *t = get_user_task_obj((uintptr_t)func_ptr);
    if (t != NULL) {
      Legion::Runtime::set_top_level_task_id(t->id);
      return Legion::Runtime::start(argc, argv);
    } else {
      return 0;
    }
  }
  
  template <typename F, typename ...Args>
  Future TaskRuntime::execute_task(F func_ptr, context &c, Args... a)
  {
    UserTask *t = get_user_task_obj((uintptr_t)func_ptr);
    if (t != NULL) {
      Future fut = t->launch_single_task(func_ptr, c, a...);
      return fut;
    } else {
      Future fut;
      return fut;
    }
  }

  template <size_t DIM, typename F, typename ...Args>
  FutureMap TaskRuntime::execute_task(F func_ptr, context &c, IdxSpace<DIM> &is, Args... a)
  {
    UserTask *t = get_user_task_obj((uintptr_t)func_ptr);
    if (t != NULL) {
      FutureMap fut = t->launch_index_task(func_ptr, c, is, a...);
      return fut;
    } else {
      FutureMap fut;
      return fut;
    }
  }

  template <size_t DIM, typename F, typename ...Args>
  FutureMap TaskRuntime::execute_task(F func_ptr, context &c, IdxSpace<DIM> &is, ArgMap argmap, Args... a)
  {
    UserTask *t = get_user_task_obj((uintptr_t)func_ptr);
    if (t != NULL) {
      FutureMap fut = t->launch_index_task(func_ptr, c, is, argmap, a...);
      return fut;
    } else {
      FutureMap fut;
      return fut;
    }
  }
}; // namespace LegionSimplified