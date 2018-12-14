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
    is = c.runtime->create_index_space(c.ctx, Legion::Domain::from_rect<DIM>(rect)); 
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
    lr = ctx.runtime->create_logical_region(ctx.ctx, ispace.is, fspace.fs);
    lr_parent = lr;
    const std::vector<field_id_t> &task_field_id_vec = fd_space.field_id_vec;
    std::vector<field_id_t>::const_iterator it; 
    for (it = task_field_id_vec.cbegin(); it != task_field_id_vec.cend(); it++) {
       printf("init inline mapping map for fid %d\n", *it);
       Base_Region<DIM> *null_ptr = nullptr;
       inline_mapping_map.insert(std::make_pair(*it, null_ptr)); 
    }
  }
  
  template <size_t DIM>
  Region<DIM>::~Region(void)
  {
    inline_mapping_map.clear();
    ctx.runtime->destroy_logical_region(ctx.ctx, lr);
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
    ip = ctx.runtime->create_partition_by_restriction(ctx.ctx, r.idx_space.is, ispace.is, dt, Legion::Domain::from_rect<DIM>(rect));
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
    printf("This shared_ptr %p new\n", this);
    region = nullptr;
    partition = nullptr;
    is_mapped = PR_NOT_MAPPED;
    task_field_vector.clear();
    accessor_map.clear();
    domain = Legion::Domain::NO_DOMAIN;
  }
  
  template <size_t DIM>
  BaseRegionImpl<DIM>::~BaseRegionImpl(void)
  {
    printf("This shared_ptr %p delete\n", this);
    region = nullptr;
    partition = nullptr;
    std::map<field_id_t, unsigned char*>::iterator it; 
    for (it = accessor_map.begin(); it != accessor_map.end(); it++) {
      if (it->second != nullptr) {
        printf("free accessor of fid %d\n", it->first);
        delete it->second;
        it->second = nullptr;
      }
    }
    accessor_map.clear();
    task_field_vector.clear();
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
    assert(this->base_region_impl->is_mapped != PR_NOT_MAPPED);
    Legion::FieldAccessor<READ_ONLY, a, DIM> *acc = get_accessor_by_fid<a>(fid);
    return (*acc)[i];
  }
  
  template <size_t DIM>
  template <typename a>
  a RO_Region<DIM>::read(Legion::Point<DIM> i)
  {
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
        printf("first time create accessor for fid %d\n", fid);
        Legion::FieldAccessor<READ_ONLY, a, DIM> *acc = new Legion::FieldAccessor<READ_ONLY, a, DIM>(this->base_region_impl->physical_region, fid);
        it->second = (unsigned char*)acc;
      }
      return (Legion::FieldAccessor<READ_ONLY, a, DIM>*)(it->second);
    } else {
      printf("can not find accessor of fid %d\n", fid);
      assert(0);
      return nullptr;
    }
  }

  template <size_t DIM>
  template< typename a>
  Legion::FieldAccessor<READ_ONLY, a, DIM>* RO_Region<DIM>::get_default_accessor(void)
  {
    assert(this->base_region_impl->task_field_vector.size() == 1);
    return get_accessor_by_fid<a>(this->base_region_impl->task_field_vector[0]);
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
    assert(this->base_region_impl->is_mapped != PR_NOT_MAPPED);
    Legion::FieldAccessor<WRITE_DISCARD, a, DIM> *acc = get_accessor_by_fid<a>(fid);
    (*acc)[i] = x; 
  }

  template <size_t DIM>
  template< typename a>
  void WD_Region<DIM>::write(Legion::Point<DIM> i, a x)
  {
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
        printf("first time create accessor for fid %d\n", fid);
        Legion::FieldAccessor<WRITE_DISCARD, a, DIM> *acc = new Legion::FieldAccessor<WRITE_DISCARD, a, DIM>(this->base_region_impl->physical_region, fid);
        it->second = (unsigned char*)acc;
      }
      return (Legion::FieldAccessor<WRITE_DISCARD, a, DIM>*)(it->second);
    } else {
      printf("can not find accessor of fid %d\n", fid);
      assert(0);
      return nullptr;
    }
  }

  template <size_t DIM>
  template< typename a>
  Legion::FieldAccessor<WRITE_DISCARD, a, DIM>* WD_Region<DIM>::get_default_accessor(void)
  {
    assert(this->base_region_impl->task_field_vector.size() == 1);
    return get_accessor_by_fid<a>(this->base_region_impl->task_field_vector[0]);
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
    assert(this->base_region_impl->is_mapped != PR_NOT_MAPPED);
    Legion::FieldAccessor<READ_WRITE, a, DIM> *acc = get_accessor_by_fid<a>(fid);
    return (*acc)[i];
  };
  
  template <size_t DIM>
  template< typename a>
  a RW_Region<DIM>::read(Legion::Point<DIM> i)
  {
    assert(this->base_region_impl->is_mapped != PR_NOT_MAPPED);
    assert(this->base_region_impl->accessor_map.size() == 1);
    Legion::FieldAccessor<READ_WRITE, a, DIM> *acc = get_default_accessor<a>();
    return (*acc)[i];
  };

  template <size_t DIM>
  template< typename a>
  void RW_Region<DIM>::write(int fid, Legion::Point<DIM> i, a x)
  {
    assert(this->base_region_impl->is_mapped != PR_NOT_MAPPED);
    Legion::FieldAccessor<READ_WRITE, a, DIM> *acc = get_accessor_by_fid<a>(fid);
    (*acc)[i] = x; 
  }

  template <size_t DIM>
  template< typename a>
  void RW_Region<DIM>::write(Legion::Point<DIM> i, a x)
  {
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
        printf("first time create accessor for fid %d\n", fid);
        Legion::FieldAccessor<READ_WRITE, a, DIM> *acc = new Legion::FieldAccessor<READ_WRITE, a, DIM>(this->base_region_impl->physical_region, fid);
        it->second = (unsigned char*)acc;
      }
      return (Legion::FieldAccessor<READ_WRITE, a, DIM>*)(it->second);
    } else {
      printf("can not find accessor of fid %d\n", fid);
      assert(0);
      return nullptr;
    }
  }

  template <size_t DIM>
  template< typename a>
  Legion::FieldAccessor<READ_WRITE, a, DIM>* RW_Region<DIM>::get_default_accessor(void)
  {
    assert(this->base_region_impl->task_field_vector.size() == 1);
    return get_accessor_by_fid<a>(this->base_region_impl->task_field_vector[0]);
  }
}; // namespace LegionSimplified