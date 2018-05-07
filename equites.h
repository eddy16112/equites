#include <stdio.h>
#include <legion.h>
#include <functional>
#include <iterator> 
#include <algorithm>
#include <array>

#include <iostream>

#define PR_NOT_MAPPED     0
#define PR_INLINE_MAPPED  1
#define PR_TASK_MAPPED    2

namespace equites { 

template <size_t ndim>  using Point = LegionRuntime::Arrays::Point<ndim>;  
template <size_t ndim>  using Rect = LegionRuntime::Arrays::Rect<ndim>;  
using LegionRuntime::Arrays::make_point;

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

/* Simple wrapper for all needed context passed to tasks */
struct context {
  const Legion::Task *task;
  Legion::Context ctx;
  Legion::Runtime *runtime;
};


template <size_t DIM>
class IdxSpace {
public:
  const context &ctx;
  Legion::IndexSpace is;
  Rect<DIM> rect;
public:
 // IdxSpace() {}
  IdxSpace(const context& c, Point<DIM> p) : ctx(c)
  {
    rect = Rect<DIM>(Point<DIM>::ZEROES(), p-Point<DIM>::ONES());
    std::cout << "ispace set rect to be from " << Point<DIM>::ZEROES() << " to " << rect.hi << std::endl; 
    is = c.runtime->create_index_space(c.ctx, Legion::Domain::from_rect<DIM>(rect)); 
  }
  
  ~IdxSpace()
  {
    ctx.runtime->destroy_index_space(ctx.ctx, is);
  }
};

class FdSpace {
public:
  const context &ctx;
  Legion::FieldSpace fs;
  Legion::FieldAllocator allocator;
  std::vector<field_id_t> field_id_vector;
public:
 // FdSpace() {}
  FdSpace(const context& c) : ctx(c)
  {
    fs = c.runtime->create_field_space(c.ctx);
    allocator = c.runtime->create_field_allocator(c.ctx, fs);
    field_id_vector.clear();
  }
  
  ~FdSpace()
  {
    ctx.runtime->destroy_field_space(ctx.ctx, fs);
  }
  
  template <typename T>
  void add_field(field_id_t fid)
  {
    allocator.allocate_field(sizeof(T),fid);
    field_id_vector.push_back(fid);
  }
};

template <size_t DIM>
class Base_Region;

template <size_t DIM>
class Region {
public:
  const context &ctx;
  IdxSpace<DIM> &idx_space; // for partition
  FdSpace &fd_space;
  Legion::LogicalRegion lr; 
  Legion::LogicalRegion lr_parent;
  std::map<int, Base_Region<DIM> *> inline_mapping_map;
public:
  //Region() {}
  Region(const context &c, IdxSpace<DIM> &ispace, FdSpace &fspace) : ctx(c), idx_space(ispace), fd_space(fspace)
  {
    lr = c.runtime->create_logical_region(c.ctx, ispace.is, fspace.fs);
    lr_parent = lr;
    std::vector<field_id_t> &task_field_id_vec = fd_space.field_id_vector;
    std::vector<field_id_t>::iterator it; 
    for (it = task_field_id_vec.begin(); it != task_field_id_vec.end(); it++) {
       printf("init inline mapping map for fid %d\n", *it);
       Base_Region<DIM> *null_ptr = NULL;
       inline_mapping_map.insert(std::make_pair(*it, null_ptr)); 
    }
  }
  
  ~Region()
  {
    inline_mapping_map.clear();
    ctx.runtime->destroy_logical_region(ctx.ctx, lr);
  }
  
  void unmap_inline_mapping(field_id_t fid)
  {
    typename std::map<int, Base_Region<DIM> *>::iterator it = inline_mapping_map.find(fid);
    if (it != inline_mapping_map.end()) {
      if (it->second != NULL) {
        printf("fid %d is already mapped, let's unmap it first\n", fid);
        Base_Region<DIM> *base_region = it->second;
        base_region->unmap_physical_region_inline();
        it->second = NULL;
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
      assert(it->second == NULL);
      it->second = base_region;
    } else {
      printf("can not find fid %d\n", fid);
      assert(0);
      return;
    }
  }
};

template <size_t DIM>
class Partition {
public:
  const context &ctx;
  Region<DIM> &region;
  Legion::IndexPartition ip;
  Legion::LogicalPartition lp;
public:
//  Partition() {}
  Partition(const context &c, Region<DIM> &r, IdxSpace<DIM> &ispace) : ctx(c), region(r)
  {
    ip = c.runtime->create_equal_partition(c.ctx, r.idx_space.is, ispace.is);
    lp = c.runtime->get_logical_partition(c.ctx, r.lr, ip);
  }
  
  ~Partition()
  {
    
  }
};

// Global id that is indexed for tasks
static Legion::TaskID globalId = 0;

// Only have one field currently.  
const static size_t OnlyField = 0; 

// would be nice to make this generic for n dimensions
/*
template <size_t ndim>
LegionRuntime::Arrays::Point<ndim> Point(array<long long,ndim> a)
  { return Point<ndim>((long long[ndim]) a.data); }
*/

// We define three helper macros. `task` defines a task, while `call` calls a
// task, and `region` creates a rw_region. These aren't strictly necessary, but
// are nice in that they hide the task context from the user and make task
// declaration a slightly easier. 
//#define task(type, name, args...) \
  type _##name (context _c, ## args); \
  _task<decltype(&_##name), _##name> name(#name); \
  type _##name (context _c, ## args) 
// call and icall simply hide the context as well
#define call(f, args...) f._call(_c, ## args) 
    //#define icall(f, args...) f._icall(_c, ## args)
// Region hides the context call as well
#define region(a, ndim, size) rw_region<a,ndim>(_c, size)
#define partition(a, ndim, ty, r, pt) Partition<a,ndim,decltype(r)>(_c, ty, r, pt)

/* A simple wrapper to avoid user-facing casting. */
    /*template <typename a> 
struct Future {
  Legion::Future fut;
  Future(Legion::Future f) { fut = f; };
  a get(){return fut.get_result<a>();};
};


template <>
struct Future<void> {
  Legion::Future fut;
  Future(Legion::Future f) { fut = f; };
  void get(){return fut.get_void_result();};
};*/
    
class Future {
public:
  Legion::Future fut;
public:  
  Future()
  {
  }
  Future(Legion::Future f)
  {
    fut = f;
  }
  template<typename T>
  T get()
  {
    return fut.get_result<T>();
  }
  void get()
  {
    return fut.get_void_result();
  }
};

class FutureMap {
public:  
  Legion::FutureMap fm;
public:
  FutureMap()
  {
  }
  FutureMap(Legion::FutureMap f) 
  { 
    fm = f; 
  }
  template<typename T, size_t DIM>
  T operator[](const Point<DIM> &point)
  { 
    Future fu(fm.get_future(Legion::DomainPoint::from_point<DIM>(point))); 
    return fu.get<T>();
  }
  template<typename T>
  T operator[](const Legion::DomainPoint &dp)
  { 
    Future fu(fm.get_future(dp)); 
    return fu.get<T>();
  }
  void wait() 
  {
    fm.wait_all_results(); 
  } // could return a container of `a`s, e.g. vector<a>
};

class ArgMap {
public:
  Legion::ArgumentMap arg_map;
public:
  ArgMap()
  {
  }
  template<typename T, size_t DIM>
  T operator[](const Point<DIM> &point)
  {
    
  }
  template<typename T>
  T operator[](const Legion::DomainPoint &dp)
  {
    
  }
  template<typename T>
  void set_point(const Legion::DomainPoint &dp, T val)
  {
    arg_map.set_point(dp, Legion::TaskArgument(&val, sizeof(T)));
  }
  template<typename T>
  static T get_arg(context &c)
  {
    return *((const T*)c.task->local_args);
  }
};

template <size_t ndim>
static Point<ndim> END(void) {
  Point<ndim> z; for(size_t i=0; i < ndim; i++) z.x[i] = -1; return z; 
}


/* regions. */
/* _region is an abstract class */
template <size_t DIM>
class Base_Region{
public:
  int is_pr_mapped;
  Legion::Domain domain;
  Region<DIM> *region;
  Partition<DIM> *partition;
  const context *ctx;
  Legion::PhysicalRegion physical_region;
  std::vector<field_id_t> task_field_vector;
  legion_privilege_mode_t pm; 
  std::map<field_id_t, void*> accessor_map;  
  
  const static legion_coherence_property_t cp = EXCLUSIVE; 
  
public:
  Base_Region()
  {
    init_parameters();
    domain = Legion::Domain::NO_DOMAIN;
    printf("base constructor\n");
  }; 
  
  Base_Region(Region<DIM> *r, std::vector<field_id_t> &task_field_id_vec) 
  {
    init_parameters();
    region = r;
    std::vector<field_id_t>::iterator it; 
    for (it = task_field_id_vec.begin(); it != task_field_id_vec.end(); it++) {
       printf("base set fid %d\n", *it);
       task_field_vector.push_back(*it); 
    }
    ctx = &(region->ctx);
    domain = Legion::Domain::from_rect<DIM>(region->idx_space.rect);
    printf("base constructor with r v\n");
  }
  
  Base_Region(Region<DIM> *r) 
  {
    init_parameters();
    region = r;
    ctx = &(region->ctx);
    std::vector<field_id_t> &task_field_id_vec = this->region->fd_space.field_id_vector;
    std::vector<field_id_t>::iterator it; 
    for (it = task_field_id_vec.begin(); it != task_field_id_vec.end(); it++) {
       printf("base set fid %d\n", *it);
       task_field_vector.push_back(*it); 
    }
    domain = Legion::Domain::from_rect<DIM>(region->idx_space.rect);
    printf("base constructor with r\n");
  }
  
  Base_Region(Partition<DIM> *par, std::vector<field_id_t> &task_field_id_vec)
  {
    init_parameters();
    partition = par;
    region = &(par->region);
    ctx = &(region->ctx);
    std::vector<field_id_t>::iterator it; 
    for (it = task_field_id_vec.begin(); it != task_field_id_vec.end(); it++) {
       printf("base set fid %d\n", *it);
       task_field_vector.push_back(*it); 
    }
    domain = Legion::Domain::from_rect<DIM>(region->idx_space.rect);
    printf("base constructor with p v\n");
  }
  
  Base_Region(Partition<DIM> *par)
  {
    init_parameters();
    partition = par;
    region = &(par->region);
    ctx = &(region->ctx);
    std::vector<field_id_t> &task_field_id_vec = this->region->fd_space.field_id_vector;
    std::vector<field_id_t>::iterator it; 
    for (it = task_field_id_vec.begin(); it != task_field_id_vec.end(); it++) {
       printf("base set fid %d\n", *it);
       task_field_vector.push_back(*it); 
    }
    domain = Legion::Domain::from_rect<DIM>(region->idx_space.rect);
    printf("base constructor with p\n");
  }
  
  ~Base_Region()
  {
    if (is_pr_mapped == PR_INLINE_MAPPED) {
      unmap_physical_region_inline();
    }
    
    std::map<field_id_t, void*>::iterator it; 
    for (it = accessor_map.begin(); it != accessor_map.end(); it++) {
      if (it->second != NULL) {
        printf("free accessor of fid %d\n", it->first);
        delete it->second;
        it->second = NULL;
      }
    }
    task_field_vector.clear();
    accessor_map.clear();
    region = NULL;
    partition = NULL;
    ctx = NULL;
  //  printf("base de-constructor\n");
  }
  
  Legion::RegionRequirement set_region_requirement_single()
  {
    Legion::RegionRequirement req(region->lr, pm, cp, region->lr_parent);
    std::vector<field_id_t>::iterator it; 
    for (it = task_field_vector.begin(); it < task_field_vector.end(); it++) {
      printf("base set RR fid %d\n", *it);
      req.add_field(*it); 
    }
    return req; 
  };
  
  Legion::RegionRequirement set_region_requirement_index()
  {
    Legion::RegionRequirement req(partition->lp, 0, pm, cp, region->lr);
    std::vector<field_id_t>::iterator it; 
    for (it = task_field_vector.begin(); it < task_field_vector.end(); it++) {
      printf("index base set RR fid %d\n", *it);
      req.add_field(*it); 
    }
    return req; 
  };
  
  void map_physical(context &c, Legion::PhysicalRegion &pr, Legion::RegionRequirement &rr)
  {
    init_parameters();
    physical_region = pr;
    std::set<field_id_t>::iterator it;
    for (it = rr.privilege_fields.begin(); it != rr.privilege_fields.end(); it++) {
      task_field_vector.push_back(*it);
      printf("map_physical rr field %d, set acc \n", *it);
      void *null_ptr = NULL;
      accessor_map.insert(std::make_pair(*it, null_ptr)); 
    }
   // ctx = &c;
    domain = c.runtime->get_index_space_domain(c.ctx, rr.region.get_index_space());
    is_pr_mapped = PR_TASK_MAPPED;
  }
  
  void map_physical_region_inline()
  {
    if (is_pr_mapped != PR_NOT_MAPPED) {
      return;
    }
    assert(ctx != NULL);
    Legion::RegionRequirement req(region->lr, pm, cp, region->lr_parent);
    std::vector<field_id_t>::iterator it; 
    for (it = task_field_vector.begin(); it < task_field_vector.end(); it++) {
      printf("base set RR fid %d\n", *it);
      req.add_field(*it);
      void *null_ptr = NULL;
      accessor_map.insert(std::make_pair(*it, null_ptr));  
      region->unmap_inline_mapping(*it);
      region->update_inline_mapping_region(this, *it);
    }
    physical_region = ctx->runtime->map_region(ctx->ctx, req);
    //domain = c.runtime->get_index_space_domain(c.ctx, req.region.get_index_space());
    is_pr_mapped = PR_INLINE_MAPPED;
  }
  
  void unmap_physical_region_inline()
  {
    if (is_pr_mapped == PR_INLINE_MAPPED) {
      assert(ctx != NULL);
      ctx->runtime->unmap_region(ctx->ctx, physical_region);
      is_pr_mapped = PR_NOT_MAPPED;
      printf("base unmap region\n");
    }
  }
  
  class iterator: public Legion::PointInDomainIterator<DIM>{
    public: 
    explicit iterator(Base_Region &r) : Legion::PointInDomainIterator<DIM>(r.domain) {} 
    bool operator()(void) {return Legion::PointInDomainIterator<DIM>::operator()();}
    iterator& operator++(void) {Legion::PointInDomainIterator<DIM>::step(); return *this; }
    iterator& operator++(int) {Legion::PointInDomainIterator<DIM>::step(); return *this; }
    const Legion::Point<DIM>& operator*(void) { return Legion::PointInDomainIterator<DIM>::operator*(); }
  };
  
private:  
  void init_parameters()
  {
    region = NULL;
    partition = NULL;
    ctx = NULL;
    is_pr_mapped = PR_NOT_MAPPED;
    task_field_vector.clear();
    accessor_map.clear();
  } 
};

// read only region

template <size_t DIM>
class RO_Region : public Base_Region<DIM> {
public:
  template< typename a>
  a read(int fid, Legion::Point<DIM> i)
  {
    assert(this->is_pr_mapped != PR_NOT_MAPPED);
    Legion::FieldAccessor<READ_ONLY, a, DIM> *acc = get_accessor_by_fid<a>(fid);
    return (*acc)[i];
  };
  
  template< typename a>
  a read(Legion::Point<DIM> i)
  {
    assert(this->is_pr_mapped != PR_NOT_MAPPED);
    Legion::FieldAccessor<READ_ONLY, a, DIM> *acc = get_default_accessor<a>();
    return (*acc)[i];
  };
  
  RO_Region() : Base_Region<DIM>()
  {
    
  }

  RO_Region(Region<DIM> *r, std::vector<field_id_t> &task_field_id_vec) : Base_Region<DIM>(r, task_field_id_vec)
  {
    init_ro_parameters();
  }
  
  RO_Region(Region<DIM> *r) : Base_Region<DIM>(r)
  {
    init_ro_parameters();
  }
  
  RO_Region(Partition<DIM> *par, std::vector<field_id_t> &task_field_id_vec) : Base_Region<DIM>(par, task_field_id_vec)
  {
    init_ro_parameters();
  }
  
  RO_Region(Partition<DIM> *par) : Base_Region<DIM>(par)
  {
    init_ro_parameters();
  }
  
  ~RO_Region()
  {
  }

private:  
  void init_ro_parameters()
  {
    this->pm = READ_ONLY;
  }
  
  template< typename a>
  Legion::FieldAccessor<READ_ONLY, a, DIM>* get_accessor_by_fid(field_id_t fid)
  {
    typename std::map<field_id_t, void*>::iterator it = this->accessor_map.find(fid);
    if (it != this->accessor_map.end()) {
      if (it->second == NULL) {
        printf("first time create accessor for fid %d\n", fid);
        Legion::FieldAccessor<READ_ONLY, a, DIM> *acc = new Legion::FieldAccessor<READ_ONLY, a, DIM>(this->physical_region, fid);
        it->second = (void*)acc;
      }
      return (Legion::FieldAccessor<READ_ONLY, a, DIM>*)(it->second);
    } else {
      printf("can not find accessor of fid %d\n", fid);
      assert(0);
      return NULL;
    }
  }
  
  template< typename a>
  Legion::FieldAccessor<READ_ONLY, a, DIM>* get_default_accessor()
  {
    assert(this->task_field_vector.size() == 1);
    return get_accessor_by_fid<a>(this->task_field_vector[0]);
  }
};

// write only region

template <size_t DIM>
class WD_Region : public Base_Region<DIM> {
public:    
  template< typename a>
  void write(int fid, Legion::Point<DIM> i, a x)
  {
    assert(this->is_pr_mapped != PR_NOT_MAPPED);
    Legion::FieldAccessor<WRITE_DISCARD, a, DIM> *acc = get_accessor_by_fid<a>(fid);
    (*acc)[i] = x; 
  }
  
  template< typename a>
  void write(Legion::Point<DIM> i, a x)
  {
    assert(this->is_pr_mapped != PR_NOT_MAPPED);
    Legion::FieldAccessor<WRITE_DISCARD, a, DIM> *acc = get_default_accessor<a>();
    (*acc)[i] = x; 
  }
  
  WD_Region() : Base_Region<DIM>()
  {
    
  }

  WD_Region(Region<DIM> *r, std::vector<field_id_t> &task_field_id_vec) : Base_Region<DIM>(r, task_field_id_vec)
  {
    init_wd_parameters();
  }
  
  WD_Region(Region<DIM> *r) : Base_Region<DIM>(r)
  {
    init_wd_parameters();
  }
  
  WD_Region(Partition<DIM> *par, std::vector<field_id_t> &task_field_id_vec) : Base_Region<DIM>(par, task_field_id_vec)
  {
    init_wd_parameters();
  }
  
  WD_Region(Partition<DIM> *par) : Base_Region<DIM>(par)
  {
    init_wd_parameters();
  }
  
  ~WD_Region()
  {
  }
  
private:  
  void init_wd_parameters()
  {
    this->pm = WRITE_DISCARD;
  }
  
  template< typename a>
  Legion::FieldAccessor<WRITE_DISCARD, a, DIM>* get_accessor_by_fid(field_id_t fid)
  {
    typename std::map<field_id_t, void*>::iterator it = this->accessor_map.find(fid);
    if (it != this->accessor_map.end()) {
      if (it->second == NULL) {
        printf("first time create accessor for fid %d\n", fid);
        Legion::FieldAccessor<WRITE_DISCARD, a, DIM> *acc = new Legion::FieldAccessor<WRITE_DISCARD, a, DIM>(this->physical_region, fid);
        it->second = (void*)acc;
      }
      return (Legion::FieldAccessor<WRITE_DISCARD, a, DIM>*)(it->second);
    } else {
      printf("can not find accessor of fid %d\n", fid);
      assert(0);
      return NULL;
    }
  }
  
  template< typename a>
  Legion::FieldAccessor<WRITE_DISCARD, a, DIM>* get_default_accessor()
  {
    assert(this->task_field_vector.size() == 1);
    return get_accessor_by_fid<a>(this->task_field_vector[0]);
  } 
};

// read-write region
template <size_t DIM>
class RW_Region : public Base_Region<DIM>{
public:
  template< typename a>
  a read(int fid, Legion::Point<DIM> i)
  {
    assert(this->is_pr_mapped != PR_NOT_MAPPED);
    Legion::FieldAccessor<READ_WRITE, a, DIM> *acc = get_accessor_by_fid<a>(fid);
    return (*acc)[i];
  };
  
  template< typename a>
  a read(Legion::Point<DIM> i)
  {
    assert(this->is_pr_mapped != PR_NOT_MAPPED);
    Legion::FieldAccessor<READ_WRITE, a, DIM> *acc = get_default_accessor<a>();
    return (*acc)[i];
  };
  
  template< typename a>
  void write(int fid, Legion::Point<DIM> i, a x)
  {
    assert(this->is_pr_mapped != PR_NOT_MAPPED);
    Legion::FieldAccessor<READ_WRITE, a, DIM> *acc = get_accessor_by_fid<a>(fid);
    (*acc)[i] = x; 
  }
  
  template< typename a>
  void write(Legion::Point<DIM> i, a x)
  {
    assert(this->is_pr_mapped != PR_NOT_MAPPED);
    Legion::FieldAccessor<READ_WRITE, a, DIM> *acc = get_default_accessor<a>();
    (*acc)[i] = x; 
  }
  
  RW_Region() : Base_Region<DIM>()
  {
    
  }

  RW_Region(Region<DIM> *r, std::vector<field_id_t> &task_field_id_vec) : Base_Region<DIM>(r, task_field_id_vec)
  {
    init_rw_parameters();
  }
  
  RW_Region(Region<DIM> *r) : Base_Region<DIM>(r)
  {
    init_rw_parameters();
  }
  
  RW_Region(Partition<DIM> *par, std::vector<field_id_t> &task_field_id_vec) : Base_Region<DIM>(par, task_field_id_vec)
  {
    init_rw_parameters();
  }
  
  RW_Region(Partition<DIM> *par) : Base_Region<DIM>(par)
  {
    init_rw_parameters();
  }
  
  ~RW_Region()
  {
  }

private:  
  void init_rw_parameters()
  {
    this->pm = READ_WRITE;
  }
  
  template< typename a>
  Legion::FieldAccessor<READ_WRITE, a, DIM>* get_accessor_by_fid(field_id_t fid)
  {
    typename std::map<field_id_t, void*>::iterator it = this->accessor_map.find(fid);
    if (it != this->accessor_map.end()) {
      if (it->second == NULL) {
        printf("first time create accessor for fid %d\n", fid);
        Legion::FieldAccessor<READ_WRITE, a, DIM> *acc = new Legion::FieldAccessor<READ_WRITE, a, DIM>(this->physical_region, fid);
        it->second = (void*)acc;
      }
      return (Legion::FieldAccessor<READ_WRITE, a, DIM>*)(it->second);
    } else {
      printf("can not find accessor of fid %d\n", fid);
      assert(0);
      return NULL;
    }
  }
  
  template< typename a>
  Legion::FieldAccessor<READ_WRITE, a, DIM>* get_default_accessor()
  {
    assert(this->task_field_vector.size() == 1);
    return get_accessor_by_fid<a>(this->task_field_vector[0]);
  }
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

template <size_t ndim, template <size_t> typename t>
inline typename std::enable_if<std::is_base_of<Base_Region<ndim>, t<ndim>>::value, int>::type
bindPhysical(context &c, std::vector<Legion::PhysicalRegion> pr, std::vector<Legion::RegionRequirement> rr, size_t i, t<ndim> *r){
  r->map_physical(c, std::ref(pr[i]), std::ref(rr[i]));    
  return i+1; 
};

// Template size_t
template<size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
bindPs(context &c, std::vector<Legion::PhysicalRegion> pr, std::vector<Legion::RegionRequirement> rr, size_t i, std::tuple<Tp...> *) { }

template<size_t I = 0, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), void>::type 
bindPs(context &c, std::vector<Legion::PhysicalRegion> pr, std::vector<Legion::RegionRequirement> rr, size_t i, std::tuple<Tp...> *t){
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
struct regTask {
  static void variant(Legion::TaskVariantRegistrar r){
    Legion::Runtime::preregister_task_variant<T, mkLegionTask<F, f>>(r);
  }
};

template <typename F, F f>
struct regTask<void, F, f> {
  static void variant(Legion::TaskVariantRegistrar r){
    Legion::Runtime::preregister_task_variant<mkLegionTask<F, f>>(r);
  }
};
/*
class TaskBase {
public:
  template <typename ...Args>
  virtual Future _call(context c, Args... a) = 0;
};*/

class UserTask {
public:
  static const UserTask NO_USER_TASK;
  Legion::TaskID id;
  std::string task_name;

public:
  UserTask(const char* name="default") : task_name(name)
  {
    id = globalId++; // still not sure about this
  }
  template <typename F, F f>
  void register_task()
  {
    typedef typename function_traits<F>::returnType RT; 
    Legion::ProcessorConstraint pc = Legion::ProcessorConstraint(Legion::Processor::LOC_PROC);
    Legion::TaskVariantRegistrar registrar(id, task_name.c_str());
    registrar.add_constraint(pc);
    regTask<RT, F, f>::variant(registrar); 
  }
  template <typename F, typename ...Args>
  Future launch_single_task(F f, context &c, Args... a)
  {
    typedef typename function_traits<F>::args argtuple;
    argtuple p = std::make_tuple(a...);
    Legion::TaskLauncher task_launcher(id, Legion::TaskArgument(&p, sizeof(p))); 
    regionArgReqs(task_launcher, p);  
    return Future(c.runtime->execute_task(c.ctx, task_launcher));
  }
  
  template <size_t DIM, typename F, typename ...Args>
  FutureMap launch_index_task(F f, context &c, IdxSpace<DIM> &ispace, Args... a){
    typedef typename function_traits<F>::args argtuple;
    argtuple p = std::make_tuple(a...);
    Legion::ArgumentMap arg_map; 
    Legion::IndexLauncher index_launcher(id, ispace.is, Legion::TaskArgument(&p, sizeof(p)), arg_map); 
    regionArgReqsIndex(index_launcher, p);  
    return FutureMap(c.runtime->execute_index_space(c.ctx, index_launcher));
  }
  
  template <size_t DIM, typename F, typename ...Args>
  FutureMap launch_index_task(F f, context &c, IdxSpace<DIM> &ispace, ArgMap argmap, Args... a){
    typedef typename function_traits<F>::args argtuple;
    argtuple p = std::make_tuple(a...);
    Legion::IndexLauncher index_launcher(id, ispace.is, Legion::TaskArgument(&p, sizeof(p)), argmap.arg_map); 
    regionArgReqsIndex(index_launcher, p);  
    return FutureMap(c.runtime->execute_index_space(c.ctx, index_launcher));
  }
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
    id = globalId++; // still not sure about this 
    Legion::TaskVariantRegistrar registrar(id, name);
    registrar.add_constraint(pc);
    regTask<RT, F, f>::variant(registrar); 
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


class TaskRuntime
{
private:
  std::map<uintptr_t, UserTask> user_task_map;
  
public:
  TaskRuntime()
  {
    
  }
  
  template <typename F, F func_ptr>
  void register_task(const char* name)
  {
    UserTask new_task(name);
    new_task.register_task<F, func_ptr>();
    user_task_map.insert(std::make_pair((uintptr_t)func_ptr, new_task)); 
  }
  
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
  UserTask* get_user_task_obj(uintptr_t func_ptr)
  {
    std::map<uintptr_t, UserTask>::iterator it = user_task_map.find(func_ptr);
    if (it != user_task_map.end()) {
      return &(it->second);
    } else {
      printf("can not find task %p\n", (void*)func_ptr);
      return NULL;
    }
  }  
};

TaskRuntime runtime;
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
} /* namespace Equites */
