#ifndef _LEGION_SIMPLIFIED_H
#define _LEGION_SIMPLIFIED_H

#include <stdio.h>
#include <legion.h>
#include <functional>
#include <iterator> 
#include <algorithm>
#include <array>

#include <iostream>
#include <memory>

#define PR_NOT_MAPPED     0
#define PR_INLINE_MAPPED  1
#define PR_TASK_MAPPED    2

namespace LegionSimplified { 
  
enum partition_type
{
  equal,
  restriction,
};

template <size_t ndim>  using Point = LegionRuntime::Arrays::Point<ndim>;  
template <size_t ndim>  using Rect = LegionRuntime::Arrays::Rect<ndim>;  
//using LegionRuntime::Arrays::make_point;

typedef Legion::FieldID field_id_t;


using coord_t = LegionRuntime::Arrays::coord_t;
inline Legion::Point<1> makepoint(coord_t x)
{
  return Legion::Point<1>(x);
}

inline Legion::Point<2> makepoint(coord_t x, coord_t y)
{
  coord_t val[2];
  val[0] = x;
  val[1] = y;
  return Legion::Point<2>(val);
}

inline Legion::Point<3> makepoint(coord_t x, coord_t y, coord_t z)
{
  coord_t val[3];
  val[0] = x;
  val[1] = y;
  val[2] = z;
  return Legion::Point<3>(val);
}

  /**
   * \struct context
   * A wrapper for all needed context passed to tasks.
   */
  struct context {
    const Legion::Task *task;
    Legion::Context ctx;
    Legion::Runtime *runtime;
  };

  /**
   * \class IdxSpace
   * A class for representing IndexSpace
   */
  template <size_t DIM>
  class IdxSpace {
  public:
    const context &ctx;
    Legion::IndexSpace is;
    Rect<DIM> rect;
  
  public:  
    IdxSpace(const context& c, Point<DIM> p);
  
    ~IdxSpace(void);
  };

  /**
   * \class FdSpace
   * A class for representing FieldSpace
   */
  class FdSpace {
  public:
    const context &ctx;
    Legion::FieldSpace fs;
    Legion::FieldAllocator allocator;
    std::vector<field_id_t> field_id_vec;
  
  public:
    FdSpace(const context& c);
  
    ~FdSpace(void);
  
    template <typename T>
    void add_field(field_id_t fid);
    
    void add_field(size_t size, field_id_t fid);
  };

  template <size_t DIM>
  class Base_Region;

  /**
   * \class Region
   * A class for representing LogicalRegion
   */
  template <size_t DIM>
  class Region {
  public:
    const context &ctx;
    const IdxSpace<DIM> &idx_space; // for partition
    const FdSpace &fd_space;
    Legion::LogicalRegion lr; 
    Legion::LogicalRegion lr_parent;
    std::map<int, Base_Region<DIM> *> inline_mapping_map;
  
  public:
    Region(IdxSpace<DIM> &ispace, FdSpace &fspace);
  
    ~Region(void);
  
    void unmap_inline_mapping(field_id_t fid)
    {
      typename std::map<int, Base_Region<DIM> *>::iterator it = inline_mapping_map.find(fid);
      if (it != inline_mapping_map.end()) {
        if (it->second != nullptr) {
          printf("fid %d is already mapped, let's unmap it first\n", fid);
          Base_Region<DIM> *base_region = it->second;
          base_region->unmap_physical_region_inline();
          it->second = nullptr;
        }
      } else {
        printf("can not find fid %d\n", fid);
        assert(0);
        return;
      }
    }
  
    void update_inline_mapping_region(Base_Region<DIM> *base_region, field_id_t fid)
    {
      typename std::map<int, Base_Region<DIM> *>::iterator it = inline_mapping_map.find(fid);
      if (it != inline_mapping_map.end()) {
        assert(it->second == nullptr);
        it->second = base_region;
      } else {
        printf("can not find fid %d\n", fid);
        assert(0);
        return;
      }
    }
  };

  /**
   * \class Partition
   * A class for representing IndexPartition and LogicalPartition
   */
  template <size_t DIM>
  class Partition {
  public:
    const context &ctx;
    const Region<DIM> &region;
    Legion::IndexPartition ip;
    Legion::LogicalPartition lp;
  
  public:
    Partition(int p_type, Region<DIM> &r, IdxSpace<DIM> &ispace);
  
    Partition(int p_type, Region<DIM> &r, IdxSpace<DIM> &ispace, Legion::DomainTransform &dt, Rect<DIM> &rect);
  
    ~Partition(void);
  };

  /**
   * \class Future
   * A class for representing Future
   */  
  class Future {
  public:
    Legion::Future fut;
    
  public:  
    Future(void)
    {
    }
    Future(Legion::Future f)
    {
      fut = f;
    }
    template<typename T>
    T get(void)
    {
      return fut.get_result<T>();
    }
    void get(void)
    {
      return fut.get_void_result();
    }
  };

  /**
   * \class FutureMap
   * A class for representing FutureMap
   */
  class FutureMap {
  public:  
    Legion::FutureMap fm;
    
  public:
    FutureMap(void)
    {
    }
    FutureMap(Legion::FutureMap f) 
    { 
      fm = f; 
    }
    template<typename T, size_t DIM>
    T get(const Point<DIM> &point)
    { 
      Future fu(fm.get_future(Legion::DomainPoint::from_point<DIM>(point))); 
      return fu.get<T>();
    }
    template<typename T>
    T get(const Point<1> &point)
    { 
      return get<T, 1>(point);
    }
    template<typename T>
    T get(const Point<2> &point)
    { 
      return get<T, 2>(point);
    }
    template<typename T>
    T get(const Point<3> &point)
    { 
      return get<T, 3>(point);
    }
    /*
    template<typename T>
    T get(const Legion::DomainPoint &dp)
    { 
      Future fu(fm.get_future(dp)); 
      return fu.get<T>();
    }*/

    void wait(void) 
    {
      fm.wait_all_results(); 
    } 
  };

class ArgMap {
public:
  Legion::ArgumentMap arg_map;
public:
  ArgMap()
  {
  }
  template<typename T, size_t DIM>
  void set_point(const Point<DIM> &point, T val)
  {
    arg_map.set_point(Legion::DomainPoint::from_point<DIM>(point), Legion::TaskArgument(&val, sizeof(T)));
  }
  template<typename T>
  void set_point(const Point<1> &point, T val)
  {
    set_point<T, 1>(point, val);
  }
  template<typename T>
  void set_point(const Point<2> &point, T val)
  {
    set_point<T, 2>(point, val);
  }
  template<typename T>
  void set_point(const Point<3> &point, T val)
  {
    set_point<T, 3>(point, val);
  }
  template<typename T>
  static T get_arg(context &c)
  {
    return *((const T*)c.task->local_args);
  }
};
/*
class Collectable {
public:
  Collectable(unsigned init = 0) : references(init) { }
public:
  inline void add_reference(unsigned cnt = 1)
  {
    __sync_add_and_fetch(&references,cnt);
    //references +=cnt;
  }
  inline bool remove_reference(unsigned cnt = 1)
  {
    unsigned prev = __sync_fetch_and_sub(&references,cnt);
    assert(prev >= cnt);
    //unsigned prev = references;
    //references-= cnt;
    return (prev == cnt);
  }
public:
unsigned int references;
}; */

  /**
   * \class BaseRegionImpl
   * A class for representing 
   */
  template <size_t DIM>
  class BaseRegionImpl {
  public:
    const Region<DIM> *region;
    const Partition<DIM> *partition;
    int is_mapped;
    Legion::Domain domain;
    Legion::PhysicalRegion physical_region;
    std::vector<field_id_t> task_field_vector;
    std::map<field_id_t, unsigned char*> accessor_map;
    
  public:
    BaseRegionImpl(void);
    
    ~BaseRegionImpl(void);
  };

/* regions. */
/* _region is an abstract class */
template <size_t DIM>
class Base_Region {
public:
  const context *ctx;
  
  //BaseRegionImpl<DIM> *base_region_impl;
  std::shared_ptr<BaseRegionImpl<DIM>> base_region_impl;
  
  legion_privilege_mode_t pm; 
  const static legion_coherence_property_t cp = EXCLUSIVE; 
  
public:
  
  Base_Region()
  {
    init_parameters();
    printf("base constructor\n");
  }; 
  
  Base_Region(const Base_Region &rhs)
  {
    init_parameters();
    this->ctx = rhs.ctx;
    this->base_region_impl = rhs.base_region_impl;
    this->pm = rhs.pm;
  }
  
  Base_Region(Region<DIM> *r, std::vector<field_id_t> &task_field_id_vec) 
  {
    init_parameters();
    base_region_impl = std::make_shared<BaseRegionImpl<DIM>>();
    base_region_impl->region = r;
    std::vector<field_id_t>::const_iterator it; 
    for (it = task_field_id_vec.cbegin(); it != task_field_id_vec.cend(); it++) {
       printf("base set fid %d\n", *it);
       base_region_impl->task_field_vector.push_back(*it); 
    }
    ctx = &(r->ctx);
    base_region_impl->domain = Legion::Domain::from_rect<DIM>(r->idx_space.rect);
    printf("base constructor with r v\n");
  }
  
  Base_Region(Region<DIM> *r) 
  {
    init_parameters();
    base_region_impl = std::make_shared<BaseRegionImpl<DIM>>();
    base_region_impl->region = r;
    ctx = &(r->ctx);
    const std::vector<field_id_t> &task_field_id_vec = base_region_impl->region->fd_space.field_id_vec;
    std::vector<field_id_t>::const_iterator it; 
    for (it = task_field_id_vec.cbegin(); it != task_field_id_vec.cend(); it++) {
       printf("base set fid %d\n", *it);
       base_region_impl->task_field_vector.push_back(*it); 
    }
    base_region_impl->domain = Legion::Domain::from_rect<DIM>(r->idx_space.rect);
    printf("base constructor with r\n");
  }
  
  Base_Region(Partition<DIM> *par, std::vector<field_id_t> &task_field_id_vec)
  {
    init_parameters();
    base_region_impl = std::make_shared<BaseRegionImpl<DIM>>();
    base_region_impl->partition = par;
    base_region_impl->region = &(par->region);
    ctx = &(base_region_impl->region->ctx);
    std::vector<field_id_t>::const_iterator it; 
    for (it = task_field_id_vec.cbegin(); it != task_field_id_vec.cend(); it++) {
       printf("base set fid %d\n", *it);
       base_region_impl->task_field_vector.push_back(*it); 
    }
    base_region_impl->domain = Legion::Domain::from_rect<DIM>(base_region_impl->region->idx_space.rect);
    printf("base constructor with p v\n");
  }
  
  Base_Region(Partition<DIM> *par)
  {
    init_parameters();
    base_region_impl = std::make_shared<BaseRegionImpl<DIM>>();
    base_region_impl->partition = par;
    base_region_impl->region = &(par->region);
    ctx = &(base_region_impl->region->ctx);
    const std::vector<field_id_t> &task_field_id_vec = base_region_impl->region->fd_space.field_id_vec;
    std::vector<field_id_t>::const_iterator it; 
    for (it = task_field_id_vec.cbegin(); it != task_field_id_vec.cend(); it++) {
       printf("base set fid %d\n", *it);
       base_region_impl->task_field_vector.push_back(*it); 
    }
    base_region_impl->domain = Legion::Domain::from_rect<DIM>(base_region_impl->region->idx_space.rect);
    printf("base constructor with p\n");
  }
  
  ~Base_Region()
  {
    /*
    if (is_pr_mapped == PR_INLINE_MAPPED) {
      unmap_physical_region_inline();
    }
    
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
  //  printf("base de-constructor\n");
  }
  
  Base_Region & operator=(const Base_Region &rhs)
  {
    init_parameters();
    this->ctx = rhs.ctx;
    this->base_region_impl = rhs.base_region_impl;
    this->pm = rhs.pm;
  }
  
  Legion::RegionRequirement set_region_requirement_single()
  {
    Legion::RegionRequirement req(base_region_impl->region->lr, pm, cp, base_region_impl->region->lr_parent);
    std::vector<field_id_t>::iterator it; 
    for (it = base_region_impl->task_field_vector.begin(); it < base_region_impl->task_field_vector.end(); it++) {
      printf("base set RR fid %d\n", *it);
      req.add_field(*it); 
    }
    return req; 
  };
  
  Legion::RegionRequirement set_region_requirement_index()
  {
    Legion::RegionRequirement req(base_region_impl->partition->lp, 0, pm, cp, base_region_impl->region->lr);
    std::vector<field_id_t>::iterator it; 
    for (it = base_region_impl->task_field_vector.begin(); it < base_region_impl->task_field_vector.end(); it++) {
      printf("index base set RR fid %d\n", *it);
      req.add_field(*it); 
    }
    return req; 
  };
  
  void map_physical_region(context &c, Legion::PhysicalRegion &pr, Legion::RegionRequirement &rr)
  {
    check_empty();
    init_parameters();
    base_region_impl = std::make_shared<BaseRegionImpl<DIM>>();
    printf("This %p, map physical new base_region_impl %p\n", this, base_region_impl.get());
    
    base_region_impl->physical_region = pr;
    std::set<field_id_t>::iterator it;
    for (it = rr.privilege_fields.begin(); it != rr.privilege_fields.end(); it++) {
      base_region_impl->task_field_vector.push_back(*it);
      printf("map_physical_region rr field %d, set acc \n", *it);
      unsigned char *null_ptr = nullptr;
      base_region_impl->accessor_map.insert(std::make_pair(*it, null_ptr)); 
    }
    ctx = &c;
    base_region_impl->domain = c.runtime->get_index_space_domain(c.ctx, rr.region.get_index_space());
    base_region_impl->is_mapped = PR_TASK_MAPPED;
  }
  
  void map_physical_region_inline()
  {
#if 0
    if (is_pr_mapped != PR_NOT_MAPPED) {
      return;
    }
    assert(ctx != NULL);
    Legion::RegionRequirement req(region->lr, pm, cp, region->lr_parent);
    std::vector<field_id_t>::iterator it; 
    for (it = task_field_vector.begin(); it < task_field_vector.end(); it++) {
      printf("base set RR fid %d\n", *it);
      req.add_field(*it);
      unsigned char *null_ptr = NULL;
      accessor_map.insert(std::make_pair(*it, null_ptr));  
      region->unmap_inline_mapping(*it);
      region->update_inline_mapping_region(this, *it);
    }
    physical_region = ctx->runtime->map_region(ctx->ctx, req);
    //domain = c.runtime->get_index_space_domain(c.ctx, req.region.get_index_space());
    is_pr_mapped = PR_INLINE_MAPPED;
  #endif
  }
  
  void unmap_physical_region_inline()
  {
#if 0
    if (is_pr_mapped == PR_INLINE_MAPPED) {
      assert(ctx != NULL);
      ctx->runtime->unmap_region(ctx->ctx, physical_region);
      is_pr_mapped = PR_NOT_MAPPED;
      printf("base unmap region\n");
    }
    #endif
  }
  
  void cleanup_reference()
  {
    ctx = nullptr;
    if (base_region_impl != nullptr) {
      auto tmp = base_region_impl;
      printf("This %p, reset base_region_impl %p, count %ld\n", this, base_region_impl.get(), base_region_impl.use_count());
      base_region_impl.reset();
      //printf("This %p, reset tmp %p, count %d\n", this, tmp.get(), tmp.use_count());
      base_region_impl = nullptr;
      printf("after nullptr %ld\n", tmp.use_count());
    }
  }
  
  class iterator: public Legion::PointInDomainIterator<DIM>{
  public: 
    iterator(bool valid = true) : is_valid(valid) {}
    explicit iterator(Base_Region &r) : Legion::PointInDomainIterator<DIM>(r.base_region_impl->domain), is_valid(true) {} 
    bool operator()(void) {return Legion::PointInDomainIterator<DIM>::operator()();}
    iterator& operator++(void) 
    {
      Legion::PointInDomainIterator<DIM>::step(); 
      if (!Legion::PointInDomainIterator<DIM>::valid()) {
        this->is_valid = false;
      }
      return *this;
    }
    iterator& operator++(int) 
    {
      Legion::PointInDomainIterator<DIM>::step(); 
      if (!Legion::PointInDomainIterator<DIM>::valid()) {
        this->is_valid = false;
      }
      return *this; 
    }
    const Legion::Point<DIM>& operator*(void) const { return Legion::PointInDomainIterator<DIM>::operator*(); }
    bool operator!=(const iterator& other) const
    {
      if (is_valid == other.is_valid) {
        return false;
      }
      const Legion::Point<DIM> my_pt = operator*();
      const Legion::Point<DIM> other_pt = other.operator*();
      return (my_pt != other_pt);
    }
  public:
    bool is_valid;
  };
  
  iterator begin()
  {
    return iterator(*this);
  }
  
  iterator end()
  {
    return iterator(false);
  }
  
private:  
  void init_parameters()
  {
    ctx = nullptr;
    base_region_impl = nullptr;
  }
  
  void check_empty()
  {
    assert(ctx == nullptr);
    assert(base_region_impl == nullptr);
      
  } 
};


  /**
   * \class RO_Region
   * A class for representing RO region
   */
  template <size_t DIM>
  class RO_Region : public Base_Region<DIM> {
  public:
    RO_Region(void);

    RO_Region(Region<DIM> *r, std::vector<field_id_t> &task_field_id_vec);
  
    RO_Region(Region<DIM> *r);
  
    RO_Region(Partition<DIM> *par, std::vector<field_id_t> &task_field_id_vec);
  
    RO_Region(Partition<DIM> *par);
  
    ~RO_Region(void);

    template< typename a>
    a read(int fid, Legion::Point<DIM> i);
  
    template< typename a>
    a read(Legion::Point<DIM> i);
    
  private:  
    void init_ro_parameters(void);
  
    template< typename a>
    Legion::FieldAccessor<READ_ONLY, a, DIM>* get_accessor_by_fid(field_id_t fid);
  
    template< typename a>
    Legion::FieldAccessor<READ_ONLY, a, DIM>* get_default_accessor(void);
  };

  /**
   * \class WD_Region
   * A class for representing WD region
   */
  template <size_t DIM>
  class WD_Region : public Base_Region<DIM> {
  public:      
    WD_Region(void);

    WD_Region(Region<DIM> *r, std::vector<field_id_t> &task_field_id_vec);
  
    WD_Region(Region<DIM> *r);
  
    WD_Region(Partition<DIM> *par, std::vector<field_id_t> &task_field_id_vec);
  
    WD_Region(Partition<DIM> *par);
  
    ~WD_Region(void);
    
    template< typename a>
    void write(int fid, Legion::Point<DIM> i, a x);
  
    template< typename a>
    void write(Legion::Point<DIM> i, a x);
  
  private:  
    void init_wd_parameters(void);
  
    template< typename a>
    Legion::FieldAccessor<WRITE_DISCARD, a, DIM>* get_accessor_by_fid(field_id_t fid);
  
    template< typename a>
    Legion::FieldAccessor<WRITE_DISCARD, a, DIM>* get_default_accessor(void);
  };

  /**
   * \class RW_Region
   * A class for representing RW region
   */
  template <size_t DIM>
  class RW_Region : public Base_Region<DIM>{
  public:  
    RW_Region(void);

    RW_Region(Region<DIM> *r, std::vector<field_id_t> &task_field_id_vec);
  
    RW_Region(Region<DIM> *r);
  
    RW_Region(Partition<DIM> *par, std::vector<field_id_t> &task_field_id_vec);
    
    RW_Region(Partition<DIM> *par);
  
    ~RW_Region(void);
    
    template< typename a>
    a read(int fid, Legion::Point<DIM> i);
  
    template< typename a>
    a read(Legion::Point<DIM> i);
  
    template< typename a>
    void write(int fid, Legion::Point<DIM> i, a x);
  
    template< typename a>
    void write(Legion::Point<DIM> i, a x);

  private:  
    void init_rw_parameters(void);
  
    template< typename a>
    Legion::FieldAccessor<READ_WRITE, a, DIM>* get_accessor_by_fid(field_id_t fid);
  
    template< typename a>
    Legion::FieldAccessor<READ_WRITE, a, DIM>* get_default_accessor(void);
  };

// Helper to apply a task to tuple 
template <typename T, typename F, size_t... Is>
inline auto apply_impl(T t, F f, std::index_sequence<Is...>){
  return f(std::get<Is>(t)...);
}

template <typename F, typename T>
inline auto apply(F f, T t){
  return apply_impl(t, f, std::make_index_sequence<std::tuple_size<T>{}>{});
}

// Helper to extract return type and argument types from function type  
template<typename T> struct function_traits {};
template<typename R, typename ...Args>
struct function_traits<R (*) (context c, Args...)> {
  static constexpr size_t nargs = sizeof...(Args);
  typedef R returnType;
  typedef std::tuple<Args...> args;
};

template<typename R, typename ...Args>
struct function_traits<R (*) (Args...)> {
  static constexpr size_t nargs = sizeof...(Args);
  typedef R returnType;
  typedef std::tuple<Args...> args;
};

// Specialization on regions
template <typename t> 
inline int bindPhysical(context &c, std::vector<Legion::PhysicalRegion> pr, std::vector<Legion::RegionRequirement> rr, size_t i, t x) { return i; }; 

template <size_t ndim, template <size_t> typename t>
inline typename std::enable_if<!std::is_base_of<Base_Region<ndim>, t<ndim>>::value, int>::type
  bindPhysical(context &c, std::vector<Legion::PhysicalRegion> pr, std::vector<Legion::RegionRequirement> rr, size_t i, t<ndim> *r){ return i; }; 

//TODO std::ref
template <size_t ndim, template <size_t> typename t>
inline typename std::enable_if<std::is_base_of<Base_Region<ndim>, t<ndim>>::value, int>::type
bindPhysical(context &c, std::vector<Legion::PhysicalRegion> pr, std::vector<Legion::RegionRequirement> rr, size_t i, t<ndim> *r){
  r->map_physical_region(c, std::ref(pr[i]), std::ref(rr[i]));    
  return i+1; 
};

// Template size_t
template<size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
bindPs(context &c, const std::vector<Legion::PhysicalRegion> &pr, const std::vector<Legion::RegionRequirement> &rr, size_t i, std::tuple<Tp...> *) { }

template<size_t I = 0, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), void>::type 
bindPs(context &c, const std::vector<Legion::PhysicalRegion> &pr, const std::vector<Legion::RegionRequirement> &rr, size_t i, std::tuple<Tp...> *t){
  bindPs<I+1>(c, pr, rr, bindPhysical(c, pr, rr, i, &std::get<I>(*t)), t); 
};

// Builds a legion task out of an arbitrary function that takes a context as a
// first argument
template <typename F, F f>
inline typename function_traits<F>::returnType  
mkLegionTask(const Legion::Task *task, const std::vector<Legion::PhysicalRegion>& pr, Legion::Context ctx, Legion::Runtime* rt) {
  typedef typename function_traits<F>::args argtuple;
  argtuple at = *(argtuple*) task->args;
  context c = { task, ctx, rt };
  size_t i = 0; 
  bindPs(c, pr, task->regions, i, &at); 
  return apply(f, std::tuple_cat(std::make_tuple(c), at));
}

// Process region requirements for function calls
template <typename t> 
inline void regionCleanUpPtr(t &r){ }; 

template <size_t ndim, template <size_t> typename t>
inline typename std::enable_if<!std::is_base_of<Base_Region<ndim>, t<ndim>>::value, void>::type
regionCleanUpPtr(t<ndim> &r){ }; 

template <size_t ndim, template <size_t> typename t>
inline typename std::enable_if<std::is_base_of<Base_Region<ndim>, t<ndim>>::value, void>::type
regionCleanUpPtr(t<ndim> &r){
  r.cleanup_reference();  
  //printf("registered region\n"); 
};

template<size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
regionCleanUp(std::tuple<Tp...> &t) {}

template<size_t I = 0, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), void>::type 
regionCleanUp(std::tuple<Tp...> &t)  {
  regionCleanUpPtr(std::get<I>(t));
  regionCleanUp<I+1>(t);  
}

// Process region requirements for function calls
template <typename t> 
inline void registerRR(Legion::TaskLauncher &l, t &r){ }; 

template <size_t ndim, template <size_t> typename t>
inline typename std::enable_if<!std::is_base_of<Base_Region<ndim>, t<ndim>>::value, void>::type
registerRR(Legion::TaskLauncher &l, t<ndim> &r){ }; 

template <size_t ndim, template <size_t> typename t>
inline typename std::enable_if<std::is_base_of<Base_Region<ndim>, t<ndim>>::value, void>::type
registerRR(Legion::TaskLauncher &l, t<ndim> &r){
  l.add_region_requirement(r.set_region_requirement_single());    
  //printf("registered region\n"); 
};

template<size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
regionArgReqs(Legion::TaskLauncher &l, std::tuple<Tp...> &t) {}

template<size_t I = 0, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), void>::type 
regionArgReqs(Legion::TaskLauncher &l, std::tuple<Tp...> &t)  {
  registerRR(l, std::get<I>(t));
  regionArgReqs<I+1>(l, t);  
}

// index
template <typename t> 
inline void registerRRIndex(Legion::IndexLauncher &l, t &r){ }; 

template <size_t ndim, template <size_t> typename t>
inline typename std::enable_if<!std::is_base_of<Base_Region<ndim>, t<ndim>>::value, void>::type
registerRRIndex(Legion::IndexLauncher &l, t<ndim> &r){ }; 

template <size_t ndim, template <size_t> typename t>
inline typename std::enable_if<std::is_base_of<Base_Region<ndim>, t<ndim>>::value, void>::type
registerRRIndex(Legion::IndexLauncher &l, t<ndim> &r){
  l.add_region_requirement(r.set_region_requirement_index());    
  //printf("registered region\n"); 
};

template<size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
regionArgReqsIndex(Legion::IndexLauncher &l, std::tuple<Tp...> &t) {}

template<size_t I = 0, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), void>::type 
regionArgReqsIndex(Legion::IndexLauncher &l, std::tuple<Tp...> &t)  {
  registerRRIndex(l, std::get<I>(t));
  regionArgReqsIndex<I+1>(l, t);  
}

  template <typename T, typename F, F f> 
  struct TaskRegistration {
    static void variant(Legion::TaskVariantRegistrar r){
      Legion::Runtime::preregister_task_variant<T, mkLegionTask<F, f>>(r);
    }
  };

  template <typename F, F f>
  struct TaskRegistration<void, F, f> {
    static void variant(Legion::TaskVariantRegistrar r){
      Legion::Runtime::preregister_task_variant<mkLegionTask<F, f>>(r);
    }
  };

  /**
   * \class UserTask
   * A class for representing user defined tasks.
   */
  class UserTask {
  public:
    static const UserTask NO_USER_TASK;
    Legion::TaskID id;
    std::string task_name;

  public:
    UserTask(const char* name);
    
    ~UserTask(void);
    
    // register task constraints
    template <typename F, F f>
    void register_task()
    {
      typedef typename function_traits<F>::returnType RT; 
      Legion::ProcessorConstraint pc = Legion::ProcessorConstraint(Legion::Processor::LOC_PROC);
      Legion::TaskVariantRegistrar registrar(id, task_name.c_str());
      registrar.add_constraint(pc);
      TaskRegistration<RT, F, f>::variant(registrar); 
    }
    
    // launch single task
    template <typename F, typename ...Args>
    Future launch_single_task(F f, context &c, Args... a)
    {
      typedef typename function_traits<F>::args argtuple;
      argtuple p = std::make_tuple(a...);
      //auto r = std::get<0>(p);
      //printf("p1 count %ld\n", r.base_region_impl.use_count());
      argtuple p2 = std::make_tuple(a...);
      //auto r2 = std::get<0>(p2);
      //printf("p2 count %ld\n", r2.base_region_impl.use_count());
      regionCleanUp(p2);
      Legion::TaskLauncher task_launcher(id, Legion::TaskArgument(&p2, sizeof(p2))); 
      regionArgReqs(task_launcher, p);  
      return Future(c.runtime->execute_task(c.ctx, task_launcher));
    }
    
    // launch index task  
    template <size_t DIM, typename F, typename ...Args>
    FutureMap launch_index_task(F f, context &c, IdxSpace<DIM> &ispace, Args... a){
      typedef typename function_traits<F>::args argtuple;
      argtuple p = std::make_tuple(a...);
      argtuple p2 = std::make_tuple(a...);
      regionCleanUp(p2);
      Legion::ArgumentMap arg_map; 
      Legion::IndexLauncher index_launcher(id, ispace.is, Legion::TaskArgument(&p2, sizeof(p2)), arg_map); 
      regionArgReqsIndex(index_launcher, p);  
      return FutureMap(c.runtime->execute_index_space(c.ctx, index_launcher));
    }
  
    // launch index task with argmap
    template <size_t DIM, typename F, typename ...Args>
    FutureMap launch_index_task(F f, context &c, IdxSpace<DIM> &ispace, ArgMap argmap, Args... a){
      typedef typename function_traits<F>::args argtuple;
      argtuple p = std::make_tuple(a...);
      argtuple p2 = std::make_tuple(a...);
      regionCleanUp(p2);
      Legion::IndexLauncher index_launcher(id, ispace.is, Legion::TaskArgument(&p2, sizeof(p2)), argmap.arg_map); 
      regionArgReqsIndex(index_launcher, p);  
      return FutureMap(c.runtime->execute_index_space(c.ctx, index_launcher));
    }
/*    
  private:
    template <typename T, typename F, F f> 
    void register_task_internal(Legion::TaskVariantRegistrar r)
    {
      Legion::Runtime::preregister_task_variant<T, mkLegionTask<F, f>>(r);
    }

    template <typename F, F f>
    void register_task_internal<void, F, f>(Legion::TaskVariantRegistrar r)
    {
        Legion::Runtime::preregister_task_variant<mkLegionTask<F, f>>(r);
    }*/
  };

template <typename F, F f>
class InternalTask {
public:
  typedef typename function_traits<F>::returnType RT; 
  typedef typename function_traits<F>::args argtuple;
  Legion::TaskID id;
  InternalTask(const char* name="default", 
        Legion::ProcessorConstraint pc = 
        Legion::ProcessorConstraint(Legion::Processor::LOC_PROC)){
  //  id = globalId++; // still not sure about this 
    Legion::TaskVariantRegistrar registrar(id, name);
    registrar.add_constraint(pc);
    TaskRegistration<RT, F, f>::variant(registrar); 
  }
  template <typename ...Args>
  Future _call(context c, Args... a){
    argtuple p = std::make_tuple(a...);
    Legion::TaskLauncher l(id, Legion::TaskArgument(&p, sizeof(p))); 
    regionArgReqs(l, p);  
    return Future(c.runtime->execute_task(c.ctx, l));
  }
  /*
  template <size_t ndim, typename ...Args>
  FutureMap<RT> _icall(context c, Point<ndim> pt, Args... a){
    argtuple p = make_tuple(a...);
    Legion::ArgumentMap arg_map; 
    Legion::IndexSpace is, pis; 
    switch(pt){
      case equal: 
        Legion::IndexPartition ip = c.runtime->create_equal_partition(c.ctx, is, pis); 
        LogicalPartition input_lp = runtime->get_logical_partition(ctx, input_lr, ip);
        Legion::IndexLauncher l(id, pis, Legion::TaskArgument(NULL, 0), arg_map); 
        regionArgReqs(l, p);  
        return FutureMap<RT>(c.runtime->execute_index_space(c.ctx, l));
    }
  }
  */
};

  /**
   * \class TaskRuntime
   * A class for runtime.
   */
  class TaskRuntime
  {
  private:
    // a map to query tasks by function ptr
    std::map<uintptr_t, UserTask> user_task_map;
  
  public:
    TaskRuntime(void);
    
    ~TaskRuntime(void);
  
    // register tasks 
    template <typename F, F func_ptr>
    void register_task(const char* name)
    {
      UserTask new_task(name);
      new_task.register_task<F, func_ptr>();
      user_task_map.insert(std::make_pair((uintptr_t)func_ptr, new_task)); 
    }
  
    // start runtime
    template <typename F>
    int start(F func_ptr, int argc, char** argv)
    { 
      UserTask *t = get_user_task_obj((uintptr_t)func_ptr);
      if (t != NULL) {
        Legion::Runtime::set_top_level_task_id(t->id);
        return Legion::Runtime::start(argc, argv);
      } else {
        return 0;
      }
    }
  
    // task launcher
    template <typename F, typename ...Args>
    Future execute_task(F func_ptr, context &c, Args... a)
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
  
    // index task launcher
    template <size_t DIM, typename F, typename ...Args>
    FutureMap execute_task(F func_ptr, context &c, IdxSpace<DIM> &is, Args... a)
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
  
    // index task launcher with argmap
    template <size_t DIM, typename F, typename ...Args>
    FutureMap execute_task(F func_ptr, context &c, IdxSpace<DIM> &is, ArgMap argmap, Args... a)
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
  
  private:
    // query task by task function ptr
    UserTask* get_user_task_obj(uintptr_t func_ptr);  
  };

  extern TaskRuntime runtime;
/*
template <void (*f) (context)>
int start(_task<void (*) (context), f> t, int argc, char** argv){ 
  Legion::Runtime::set_top_level_task_id(t.id);
  return Legion::Runtime::start(argc, argv);
};

template <void (*f) (context, int, char**)>
int start(_task<void (*) (context, int, char**), f> t, int argc, char** argv){ 
  Legion::Runtime::set_top_level_task_id(t.id);
  return Legion::Runtime::start(argc, argv);
};*/


/*
template <size_t ndim> 
void _fill(context _c, w_region<ndim> r, a v); 
template <typename a, size_t ndim> 
InternalTask<decltype(&_fill<a,ndim>), _fill<a,ndim>> fill("fill"); 
template <typename a, size_t ndim>
void _fill(context _c, w_region<a, ndim> r, a v){
  for(auto i : r) r.write(i, v);
}

template <typename a, size_t ndim> 
void _print(context _c, r_region<a, ndim> r); 
template <typename a, size_t ndim> 
InternalTask<decltype(&_print<a,ndim>), _print<a,ndim>> print("print"); 
template <typename a, size_t ndim>
void _print(context _c, r_region<a, ndim> r){
  for(auto i : r){
    std::cout << i << ": " << r.read(i) << std::endl; 
  }
}
*/
/*
template <typename a, size_t ndim> 
void _copy(context _c, r_region<a, ndim> r, w_region<a, ndim> w); 
template <typename a, size_t ndim>
_task<decltype(&_copy<a,ndim>), _copy<a,ndim>> copy("copy"); 
template <typename a, size_t ndim>
void _copy(context _c, r_region<a, ndim> r, w_region<a, ndim> w){
  for(auto i : r){
    w.write(i, r.read(i)); 
  }
}
*/
}; // namespace LegionSimplified

#include "legion_simplified.inl"

#endif // _LEGION_SIMPLIFIED_H
