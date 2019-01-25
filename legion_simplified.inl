#include "legion_simplified.h"

namespace LegionSimplified {

  /////////////////////////////////////////////////////////////
  // IdxSpace
  /////////////////////////////////////////////////////////////
  
  //----------------------------------public-------------------------------------
  template <size_t DIM>
  IdxSpace<DIM>::IdxSpace(const context& c, Point<DIM> p) : ctx(c)
  {
    rect = Rect<DIM>(Point<DIM>::ZEROES(), p-Point<DIM>::ONES());
    std::cout << "ispace set rect to be from " << Point<DIM>::ZEROES() 
      << " to " << rect.hi << std::endl; 
    is = c.runtime->create_index_space(c.ctx, rect); 
  }
  
  template <size_t DIM>
  IdxSpace<DIM>::~IdxSpace(void)
  {
    ctx.runtime->destroy_index_space(ctx.ctx, is);
  }
  
  /////////////////////////////////////////////////////////////
  // FdSpace
  /////////////////////////////////////////////////////////////
  
  //----------------------------------public-------------------------------------
  template <typename T>
  void FdSpace::add_field(field_id_t fid)
  {
    allocator.allocate_field(sizeof(T),fid);
    field_id_vec.push_back(fid);
  }
  
  /////////////////////////////////////////////////////////////
  // Region
  /////////////////////////////////////////////////////////////
  
  //----------------------------------public-------------------------------------
  template <size_t DIM>
  Region<DIM>::Region(IdxSpace<DIM> &ispace, FdSpace &fspace) : 
    ctx(ispace.ctx), idx_space(ispace), fd_space(fspace)
  {
    DEBUG_PRINT((4, "Region constructor %p\n", this));
    lr = ctx.runtime->create_logical_region(ctx.ctx, ispace.is, fspace.fs);
    lr_parent = lr;
    const std::vector<field_id_t> &task_field_id_vec = fd_space.field_id_vec;
    std::vector<field_id_t>::const_iterator it; 
    for (it = task_field_id_vec.cbegin(); it != task_field_id_vec.cend(); it++) {
       DEBUG_PRINT((6, "Init inline_mapping_map for fid %d\n", *it));
       Base_Region<DIM> *null_ptr = nullptr;
       inline_mapping_map.insert(std::make_pair(*it, null_ptr)); 
    }
  }
  
  template <size_t DIM>
  Region<DIM>::~Region(void)
  {
    DEBUG_PRINT((4, "Region destructor %p\n", this));
    typename std::map<field_id_t, Base_Region<DIM> *>::iterator it;
    for (it = inline_mapping_map.begin(); it != inline_mapping_map.end(); it++) {
      if (it->second != nullptr) {
        assert(0);
      }
    }
    inline_mapping_map.clear();
    ctx.runtime->destroy_logical_region(ctx.ctx, lr);
  }
  
  template <size_t DIM>
  void Region<DIM>:: update_inline_mapping_map(field_id_t fid, Base_Region<DIM> *new_base_region) const
  {
    typename std::map<field_id_t, Base_Region<DIM> *>::iterator it = inline_mapping_map.find(fid);
    DEBUG_PRINT((6, "Region %p, update_inline_mapping_map for fid %d\n", this, fid));
    if (it != inline_mapping_map.end()) {
      if (it->second != nullptr) {
        DEBUG_PRINT((6, "Region %p, fid %d is already mapped, let's unmap it first, and update the base_region from %p to %p \n", this, fid, it->second, new_base_region));
        Base_Region<DIM> *base_region = it->second;
        base_region->unmap_physical_region_inline();
        it->second = new_base_region;
      } else {
        it->second = new_base_region;
      }
    } else {
      DEBUG_PRINT((0, "can not find fid %d\n", fid));
      assert(0);
      return;
    }
  }
  
  template <size_t DIM>
  void Region<DIM>:: remove_inline_mapping_map(Base_Region<DIM> *old_base_region) const
  {
    typename std::map<field_id_t, Base_Region<DIM> *>::iterator it;
    for (it = inline_mapping_map.begin(); it != inline_mapping_map.end(); it++) {
      if (it->second == old_base_region) {
        DEBUG_PRINT((6, "Region %p, find fid %ld, base_region %p\n", this, it->first, it->second));
        it->second = nullptr;
      }
    }
  }
  
  /////////////////////////////////////////////////////////////
  // Partition
  /////////////////////////////////////////////////////////////
  
  //----------------------------------public-------------------------------------
  template <size_t DIM>
  Partition<DIM>::Partition(int p_type, Region<DIM> &r, IdxSpace<DIM> &ispace) : 
    ctx(r.ctx), region(r)
  {
    ip = ctx.runtime->create_equal_partition(ctx.ctx, r.idx_space.is, ispace.is);
    lp = ctx.runtime->get_logical_partition(ctx.ctx, r.lr, ip);
  }
  
  template <size_t DIM>
  Partition<DIM>::Partition(int p_type, Region<DIM> &r, IdxSpace<DIM> &ispace, 
    Legion::DomainTransform &dt, Rect<DIM> &rect) : 
    ctx(r.ctx), region(r)
  {
    ip = ctx.runtime->create_partition_by_restriction(ctx.ctx, r.idx_space.is, ispace.is, dt, rect);
    lp = ctx.runtime->get_logical_partition(ctx.ctx, r.lr, ip);
  }
  
  template <size_t DIM>
  Partition<DIM>::~Partition(void)
  {
  }
  
  /////////////////////////////////////////////////////////////
  // BaseRegionImpl
  ///////////////////////////////////////////////////////////// 
  
  //----------------------------------public-------------------------------------
  template <size_t DIM>
  BaseRegionImpl<DIM>::BaseRegionImpl(void) : 
    region(nullptr), partition(nullptr), is_mapped(PR_NOT_MAPPED)
  {
    DEBUG_PRINT((2, "BaseRegionImpl(shared_ptr) constructor %p\n", this));
    region = nullptr;
    partition = nullptr;
    is_mapped = PR_NOT_MAPPED;
    field_id_vector.clear();
    accessor_map.clear();
    domain = Legion::Domain::NO_DOMAIN;
  }
  
  template <size_t DIM>
  BaseRegionImpl<DIM>::~BaseRegionImpl(void)
  {
    DEBUG_PRINT((2, "BaseRegionImpl(shared_ptr) destructor %p\n", this));
    region = nullptr;
    partition = nullptr;
    std::map<field_id_t, unsigned char*>::iterator it; 
    for (it = accessor_map.begin(); it != accessor_map.end(); it++) {
      if (it->second != nullptr) {
        DEBUG_PRINT((4, "BaseRegionImpl %p, free accessor of fid %d\n", this, it->first));
        delete it->second;
        it->second = nullptr;
      }
    }
    accessor_map.clear();
    field_id_vector.clear();
  }
  
  template <size_t DIM>
  void BaseRegionImpl<DIM>::init_accessor_map()
  {
    assert(field_id_vector.size() != 0);
    std::vector<field_id_t>::iterator it; 
    for (it = field_id_vector.begin(); it < field_id_vector.end(); it++) {
      DEBUG_PRINT((4, "BaseRegionImpl %p, init_accessor_map for fid %d\n", this, *it));
      unsigned char *null_ptr = NULL;
      accessor_map.insert(std::make_pair(*it, null_ptr));
    }
  }
  
  
  /////////////////////////////////////////////////////////////
  // Base_Region 
  /////////////////////////////////////////////////////////////
  
  //----------------------------------public-------------------------------------
  template <size_t DIM>
  Base_Region<DIM>::Base_Region()
  {
    init_parameters();
    DEBUG_PRINT((4, "Base_Region empty constructor %p\n", this));
  } 

  template <size_t DIM>
  Base_Region<DIM>::Base_Region(const Base_Region &rhs)
  {
    DEBUG_PRINT((8, "Base_Region copy constructor %p\n", this));
    init_parameters();
    this->ctx = rhs.ctx;
    this->base_region_impl = rhs.base_region_impl;
    this->pm = rhs.pm;
  }

  template <size_t DIM>
  Base_Region<DIM>::Base_Region(Region<DIM> *r, std::vector<field_id_t> &task_field_id_vec) 
  {
    DEBUG_PRINT((4, "Base_Region Region/field constructor %p\n", this));
    init_parameters();
    base_region_impl = std::make_shared<BaseRegionImpl<DIM>>();
    base_region_impl->region = r;
    std::vector<field_id_t>::const_iterator it; 
    for (it = task_field_id_vec.cbegin(); it != task_field_id_vec.cend(); it++) {
       printf("base set fid %d\n", *it);
       base_region_impl->field_id_vector.push_back(*it); 
    }
    ctx = &(r->ctx);
    base_region_impl->domain = Legion::Domain(r->idx_space.rect);
  }

  template <size_t DIM>
  Base_Region<DIM>::Base_Region(Region<DIM> *r) 
  {
    DEBUG_PRINT((4, "Base_Region Region constructor %p\n", this));
    init_parameters();
    base_region_impl = std::make_shared<BaseRegionImpl<DIM>>();
    base_region_impl->region = r;
    ctx = &(r->ctx);
    const std::vector<field_id_t> &task_field_id_vec = base_region_impl->region->fd_space.field_id_vec;
    std::vector<field_id_t>::const_iterator it; 
    for (it = task_field_id_vec.cbegin(); it != task_field_id_vec.cend(); it++) {
       DEBUG_PRINT((6, "Base_Region %p, set fid %d\n", this, *it));
       base_region_impl->field_id_vector.push_back(*it); 
    }
    base_region_impl->domain = Legion::Domain(r->idx_space.rect);
  }

  template <size_t DIM>
  Base_Region<DIM>::Base_Region(Partition<DIM> *par, std::vector<field_id_t> &task_field_id_vec)
  {
    DEBUG_PRINT((4, "Base_Region Partition/field constructor %p\n", this));
    init_parameters();
    base_region_impl = std::make_shared<BaseRegionImpl<DIM>>();
    base_region_impl->partition = par;
    base_region_impl->region = &(par->region);
    ctx = &(base_region_impl->region->ctx);
    std::vector<field_id_t>::const_iterator it; 
    for (it = task_field_id_vec.cbegin(); it != task_field_id_vec.cend(); it++) {
       DEBUG_PRINT((6, "Base_Region %p, set fid %d\n", this, *it));
       base_region_impl->field_id_vector.push_back(*it); 
    }
    base_region_impl->domain = Legion::Domain(base_region_impl->region->idx_space.rect);
  }

  template <size_t DIM>
  Base_Region<DIM>::Base_Region(Partition<DIM> *par)
  {
    DEBUG_PRINT((4, "Base_Region Partition constructor %p\n", this));
    init_parameters();
    base_region_impl = std::make_shared<BaseRegionImpl<DIM>>();
    base_region_impl->partition = par;
    base_region_impl->region = &(par->region);
    ctx = &(base_region_impl->region->ctx);
    const std::vector<field_id_t> &task_field_id_vec = base_region_impl->region->fd_space.field_id_vec;
    std::vector<field_id_t>::const_iterator it; 
    for (it = task_field_id_vec.cbegin(); it != task_field_id_vec.cend(); it++) {
       DEBUG_PRINT((6, "Base_Region %p, set fid %d\n", this, *it));
       base_region_impl->field_id_vector.push_back(*it); 
    }
    base_region_impl->domain = Legion::Domain(base_region_impl->region->idx_space.rect);
  }
  
  template <size_t DIM>
  Base_Region<DIM>::~Base_Region(void)
  {
    if (base_region_impl != nullptr) {
      if (base_region_impl->is_mapped == PR_INLINE_MAPPED) {
        base_region_impl->region->remove_inline_mapping_map(this);
        unmap_physical_region_inline();
      }
    }
    /*
    std::map<field_id_t, unsigned char*>::iterator it; 
    for (it = accessor_map.begin(); it != accessor_map.end(); it++) {
      if (it->second != NULL) {
        printf("free accessor of fid %d\n", it->first);
        delete it->second;
        it->second = NULL;
      }
    }*/

    ctx = nullptr;
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
  Legion::RegionRequirement Base_Region<DIM>::set_region_requirement_single()
  {
    Legion::RegionRequirement req(base_region_impl->region->lr, pm, cp, base_region_impl->region->lr_parent);
    std::vector<field_id_t>::iterator it; 
    for (it = base_region_impl->field_id_vector.begin(); it < base_region_impl->field_id_vector.end(); it++) {
      DEBUG_PRINT((4, "Base_Region %p, set RR fid %d for single launcher\n", this, *it));
      req.add_field(*it); 
    }
    return req; 
  }

  template <size_t DIM>
  Legion::RegionRequirement Base_Region<DIM>::set_region_requirement_index()
  {
    Legion::RegionRequirement req(base_region_impl->partition->lp, 0, pm, cp, base_region_impl->region->lr);
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
    base_region_impl = std::make_shared<BaseRegionImpl<DIM>>();
    DEBUG_PRINT((4, "Base_Region %p, map_physical_region, BaseRegionImpl shared_ptr %p\n", this, base_region_impl.get()));
  
    base_region_impl->physical_region = pr;
    std::set<field_id_t>::iterator it;
    for (it = rr.privilege_fields.begin(); it != rr.privilege_fields.end(); it++) {
      base_region_impl->field_id_vector.push_back(*it);
      DEBUG_PRINT((4, "Base_Region %p, map_physical_region rr field %d, set accessor \n", this, *it));
      unsigned char *null_ptr = nullptr;
      base_region_impl->accessor_map.insert(std::make_pair(*it, null_ptr)); 
    }
    ctx = &c;
    base_region_impl->domain = c.runtime->get_index_space_domain(c.ctx, rr.region.get_index_space());
    base_region_impl->is_mapped = PR_TASK_MAPPED;
  }
  
  template <size_t DIM>
  void Base_Region<DIM>::map_physical_region_inline_with_auto_unmap()
  {
    if (base_region_impl->is_mapped != PR_NOT_MAPPED) {
      return;
    }
    DEBUG_PRINT((4, "Base_Region %p, map_physical_region_inline, BaseRegionImpl shared_ptr %p\n", this, base_region_impl.get()));
    assert(ctx != NULL);
    Legion::RegionRequirement req(base_region_impl->region->lr, pm, cp, base_region_impl->region->lr_parent);
    std::vector<field_id_t>::iterator it; 
    for (it = base_region_impl->field_id_vector.begin(); it < base_region_impl->field_id_vector.end(); it++) {
      DEBUG_PRINT((4, "Base_Region %p, inline map fid %d\n", this, *it));
      req.add_field(*it);
      unsigned char *null_ptr = NULL;
      base_region_impl->accessor_map.insert(std::make_pair(*it, null_ptr));  
      base_region_impl->region->update_inline_mapping_map(*it, this);
    }
    base_region_impl->physical_region = ctx->runtime->map_region(ctx->ctx, req);
    base_region_impl->domain = ctx->runtime->get_index_space_domain(ctx->ctx, req.region.get_index_space());
    base_region_impl->is_mapped = PR_INLINE_MAPPED;
  }
  
  template <size_t DIM>
  void Base_Region<DIM>::map_physical_region_inline()
  {
    if (base_region_impl->is_mapped != PR_NOT_MAPPED) {
      return;
    }
    DEBUG_PRINT((4, "Base_Region %p, map_physical_region_inline, BaseRegionImpl shared_ptr %p\n", this, base_region_impl.get()));
    assert(ctx != NULL);
    Legion::RegionRequirement req(base_region_impl->region->lr, pm, cp, base_region_impl->region->lr_parent);
    std::vector<field_id_t>::iterator it; 
    for (it = base_region_impl->field_id_vector.begin(); it < base_region_impl->field_id_vector.end(); it++) {
      DEBUG_PRINT((4, "Base_Region %p, inline map for fid %d\n", this, *it));
      req.add_field(*it);
      unsigned char *null_ptr = NULL;
      base_region_impl->accessor_map.insert(std::make_pair(*it, null_ptr));  
    }
    base_region_impl->physical_region = ctx->runtime->map_region(ctx->ctx, req);
    base_region_impl->domain = ctx->runtime->get_index_space_domain(ctx->ctx, req.region.get_index_space());
    base_region_impl->is_mapped = PR_INLINE_MAPPED;
  }
  
  template <size_t DIM>
  void Base_Region<DIM>::unmap_physical_region_inline()
  {
    if (base_region_impl->is_mapped == PR_INLINE_MAPPED) {
      assert(ctx != NULL);
      ctx->runtime->unmap_region(ctx->ctx, base_region_impl->physical_region);
      base_region_impl->is_mapped = PR_NOT_MAPPED;
      DEBUG_PRINT((4, "Base_Region %p, unmap region inline\n", this));
    }
  }
  
  template <size_t DIM>
  void Base_Region<DIM>::cleanup_reference()
  {
    ctx = nullptr;
    if (base_region_impl != nullptr) {
      auto tmp = base_region_impl;
      DEBUG_PRINT((4, "Base_Region %p, reset base_region_impl %p, count %ld\n", this, base_region_impl.get(), base_region_impl.use_count()));
      base_region_impl.reset();
      DEBUG_PRINT((4, "Base_Region %p, reset tmp %p, count %ld\n", this, tmp.get(), tmp.use_count()));
      base_region_impl = nullptr;
      DEBUG_PRINT((4, "after nullptr %ld\n", tmp.use_count()));
    }
  }
  
  template <size_t DIM>
  void Base_Region<DIM>::if_mapped()
  { 
    // not mapped, do inline mapping
    if (base_region_impl->is_mapped == PR_NOT_MAPPED) {
      map_physical_region_inline_with_auto_unmap();
    }
  }
  
  //----------------------------------private-------------------------------------
  template <size_t DIM>
  void Base_Region<DIM>::init_parameters()
  {
    ctx = nullptr;
    base_region_impl = nullptr;
  }

  template <size_t DIM>
  void Base_Region<DIM>::check_empty()
  {
    assert(ctx == nullptr);
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
  RO_Region<DIM>::RO_Region(Region<DIM> *r, std::vector<field_id_t> &task_field_id_vec) 
    : Base_Region<DIM>(r, task_field_id_vec)
  {
    init_ro_parameters();
  }

  template <size_t DIM>
  RO_Region<DIM>::RO_Region(Region<DIM> *r) 
    : Base_Region<DIM>(r)
  {
    init_ro_parameters();
  }

  template <size_t DIM>
  RO_Region<DIM>::RO_Region(Partition<DIM> *par, std::vector<field_id_t> &task_field_id_vec) 
    : Base_Region<DIM>(par, task_field_id_vec)
  {
    init_ro_parameters();
  }

  template <size_t DIM>
  RO_Region<DIM>::RO_Region(Partition<DIM> *par) 
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
  
  //----------------------------------private-------------------------------------
  template <size_t DIM>
  void RO_Region<DIM>::init_ro_parameters(void)
  {
    this->pm = READ_ONLY;
  }
  
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
  WD_Region<DIM>::WD_Region(Region<DIM> *r, std::vector<field_id_t> &task_field_id_vec) 
    : Base_Region<DIM>(r, task_field_id_vec)
  {
    init_wd_parameters();
  }

  template <size_t DIM>
  WD_Region<DIM>::WD_Region(Region<DIM> *r) : Base_Region<DIM>(r)
  {
    init_wd_parameters();
  }

  template <size_t DIM>
  WD_Region<DIM>::WD_Region(Partition<DIM> *par, std::vector<field_id_t> &task_field_id_vec) 
    : Base_Region<DIM>(par, task_field_id_vec)
  {
    init_wd_parameters();
  }

  template <size_t DIM>
  WD_Region<DIM>::WD_Region(Partition<DIM> *par) 
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
  
  //----------------------------------private-------------------------------------
  template <size_t DIM>
  void WD_Region<DIM>::init_wd_parameters(void)
  {
    this->pm = WRITE_DISCARD;
  }

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
  RW_Region<DIM>::RW_Region(Region<DIM> *r, std::vector<field_id_t> &task_field_id_vec) 
    : Base_Region<DIM>(r, task_field_id_vec)
  {
    init_rw_parameters();
  }

  template <size_t DIM>
  RW_Region<DIM>::RW_Region(Region<DIM> *r) 
    : Base_Region<DIM>(r)
  {
    init_rw_parameters();
  }

  template <size_t DIM>
  RW_Region<DIM>::RW_Region(Partition<DIM> *par, std::vector<field_id_t> &task_field_id_vec) 
    : Base_Region<DIM>(par, task_field_id_vec)
  {
    init_rw_parameters();
  }

  template <size_t DIM>
  RW_Region<DIM>::RW_Region(Partition<DIM> *par) 
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
  
  //----------------------------------private-------------------------------------
  template <size_t DIM>
  void RW_Region<DIM>::init_rw_parameters(void)
  {
    this->pm = READ_WRITE;
  }

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
  // UserTask 
  /////////////////////////////////////////////////////////////
  
  //----------------------------------public-------------------------------------
  template <typename F, F f>
  void UserTask::register_task(void)
  {
    typedef typename function_traits<F>::returnType RT; 
    Legion::ProcessorConstraint pc = Legion::ProcessorConstraint(Legion::Processor::LOC_PROC);
    Legion::TaskVariantRegistrar registrar(id, task_name.c_str());
    registrar.add_constraint(pc);
    TaskRegistration<RT, F, f>::variant(registrar); 
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
    Legion::IndexLauncher index_launcher(id, ispace.is, Legion::TaskArgument(&p2, sizeof(p2)), arg_map); 
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
    Legion::IndexLauncher index_launcher(id, ispace.is, Legion::TaskArgument(&p2, sizeof(p2)), argmap.arg_map); 
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
    new_task.register_task<F, func_ptr>();
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