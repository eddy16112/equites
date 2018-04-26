#include <stdio.h>
#include <legion.h>
#include <functional>
#include <iterator> 
#include <algorithm>
#include <array>

#include <iostream>

namespace equites { 

template <size_t ndim>  using Point = LegionRuntime::Arrays::Point<ndim>;  
template <size_t ndim>  using Rect = LegionRuntime::Arrays::Rect<ndim>;  
using LegionRuntime::Arrays::make_point;


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
  Legion::IndexSpace idx_space;
  Rect<DIM> rect;
public:
  IdxSpace(const context& c, Point<DIM> p)
  {
    rect = Rect<DIM>(Point<DIM>::ZEROES(), p-Point<DIM>::ONES());
    std::cout << "ispace set rect to be from " << Point<DIM>::ZEROES() << " to " << rect.hi << std::endl; 
    idx_space = c.runtime->create_index_space(c.ctx, Legion::Domain::from_rect<DIM>(rect)); 
  }
};

class FdSpace {
public:
  Legion::FieldSpace fd_space;
  Legion::FieldAllocator allocator;
public:
  FdSpace(const context& c)
  {
    fd_space = c.runtime->create_field_space(c.ctx);
    allocator = c.runtime->create_field_allocator(c.ctx, fd_space);
  }
  template <typename T>
  void add_field(int fid)
  {
    allocator.allocate_field(sizeof(T),fid);
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

/*
template <typename a> 
struct FutureMap {
  Legion::FutureMap fut;
  FutureMap(Legion::FutureMap f) { fut = f; };
  inline Future<a> operator[](const Legion::DomainPoint &point)
    { return Future<a>(fut.get_future(point)); }
  void wait() {fut.wait_all_results(); } // could return a container of `a`s, e.g. vector<a>
};*/

template <size_t ndim>
static Point<ndim> END(void) {
  Point<ndim> z; for(size_t i=0; i < ndim; i++) z.x[i] = -1; return z; 
}

#if 0
/* regions. */
/* _region is an abstract class */
template <typename a, size_t ndim>
struct _region{
  _region(){}; 
  _region(const Rect<ndim> r) : rect(r) {}; 
  class iterator: public std::iterator <std::input_iterator_tag, a, Point<ndim>> {
    public: 
    Point<ndim> pt; 
    Rect<ndim> r; 
    explicit iterator(const Rect<ndim> rect, Point<ndim> p) : pt(p), r(rect) {} ; 
    void step() {
      for(size_t i=0; i<ndim; i++){
        if(++pt.x[i] <= r.hi.x[i]) return; 
        pt.x[i] = r.lo.x[i]; 
      }
      // If we fall through, set to pre-defined end() value
      for(size_t i=0; i<ndim; i++){
        pt.x[i] = -1; 
      }
    }
    iterator& operator++() {step(); return *this; }
    iterator operator++(int) {iterator retval = *this; ++(*this); return retval; }
    bool operator==(iterator other) const { return pt == other.pt; }
    bool operator!=(iterator other) const { return !(*this == other); }
    Point<ndim> operator*() const { return pt; }
  }; 

  Legion::RegionRequirement rr(){
    Legion::RegionRequirement req(this->l, this->pm, this->cp, this->parent); 
    req.add_field(OnlyField); 
    return req; 
  };

  void setPhysical(context &c, Legion::PhysicalRegion &p){
    this->p = p;
    this->acc = Legion::FieldAccessor<NO_ACCESS, a, ndim>(p, OnlyField);
  }

  Legion::Domain domain;
  Legion::PointInDomainIterator<ndim> domain_iterator;
  Legion::FieldAccessor<NO_ACCESS, a, ndim> acc; 
  Legion::PhysicalRegion p;
  Legion::LogicalRegion l; 
  Legion::IndexSpace is; 
  Legion::LogicalRegion parent; 
  Rect<ndim> rect; 
  const static legion_privilege_mode_t pm = NO_ACCESS;  
  const static legion_coherence_property_t cp = EXCLUSIVE; 
  iterator begin() { return iterator(rect, rect.lo); }
  iterator end() { return iterator(rect, END<ndim>()); }
};
#else
/* regions. */
/* _region is an abstract class */
template <typename a, size_t ndim>
struct _region{
  _region(){}; 
  _region(const Rect<ndim> r) : rect(r) {}; 
  class iterator: public Legion::PointInDomainIterator<ndim>{
    public: 
    explicit iterator(Legion::Domain d) : Legion::PointInDomainIterator<ndim>(d) {} ; 
    bool operator()(void) {return Legion::PointInDomainIterator<ndim>::operator()();}
    iterator& operator++(void) {Legion::PointInDomainIterator<ndim>::step(); return *this; }
    iterator& operator++(int) {Legion::PointInDomainIterator<ndim>::step(); return *this; }
    const Legion::Point<ndim>& operator*(void) { return Legion::PointInDomainIterator<ndim>::operator*(); }
  }; 

  Legion::RegionRequirement rr(){
    Legion::RegionRequirement req(this->l, this->pm, this->cp, this->parent); 
    req.add_field(OnlyField); 
    return req; 
  };

  void setPhysical(context &c, Legion::PhysicalRegion &p){
    this->p = p;
    this->acc = Legion::FieldAccessor<NO_ACCESS, a, ndim>(p, OnlyField);
  }

  Legion::Domain domain;
  Legion::FieldAccessor<NO_ACCESS, a, ndim> acc; 
  Legion::PhysicalRegion p;
  Legion::LogicalRegion l; 
  Legion::IndexSpace is; 
  Legion::LogicalRegion parent; 
  Rect<ndim> rect; 
  const static legion_privilege_mode_t pm = NO_ACCESS;  
  const static legion_coherence_property_t cp = EXCLUSIVE; 
};
#endif

// read only region
template <typename a, size_t ndim>
struct r_region : virtual _region<a, ndim> {
  r_region(){}; 
  Legion::RegionRequirement rr(){
    Legion::RegionRequirement req(this->l, this->pm, this->cp, this->parent); 
    req.add_field(OnlyField); 
    return req; 
  };
  a read(Point<ndim> i){
    return this->acc.read(Legion::DomainPoint::from_point<ndim>(i));
  };
  void setPhysical(context &c, Legion::PhysicalRegion &p){
    this->p = p;
    this->acc = Legion::FieldAccessor<READ_ONLY, a, ndim>(p, OnlyField);
  }

  Legion::FieldAccessor<READ_ONLY, a, ndim> acc; 
  const static legion_privilege_mode_t pm = READ_ONLY;  
};

// write only region
template <typename a, size_t ndim>
struct w_region : virtual _region<a, ndim> {
  w_region(){}; 
  Legion::RegionRequirement rr(){
    Legion::RegionRequirement req(this->l, this->pm, this->cp, this->parent); 
    req.add_field(OnlyField); 
    return req; 
  };
  void write(Point<ndim> i, a x){
    this->acc.write(Legion::DomainPoint::from_point<ndim>(i), x); 
  }
  void setPhysical(context &c, Legion::PhysicalRegion &p){
    this->p = p;
    this->acc = Legion::FieldAccessor<READ_WRITE, a, ndim>(p, OnlyField);
  }

  Legion::FieldAccessor<READ_WRITE, a, ndim> acc; 
  const static legion_privilege_mode_t pm = READ_WRITE;  
};

// read-write region
template <typename a, size_t ndim>
struct rw_region : virtual r_region<a, ndim>, virtual w_region<a, ndim> {
  Legion::RegionRequirement rr()
  {
    Legion::RegionRequirement req(this->l, this->pm, this->cp, this->parent); 
    for (int i = 0; i < nb_fields; i++) {
      printf("rw set RR fid %d\n", task_fields[i]);
      req.add_field(task_fields[i]); 
    }
    return req; 
  };
  
  a read(Point<ndim> i, int fid)
  {
    return acc_array[fid].read(Legion::DomainPoint::from_point<ndim>(i));
  };
  
  a read(Legion::Point<ndim> i, int fid)
  {
    return (acc_array[fid])[i];
  };
  
  void write(Point<ndim> i, int fid, a x)
  {
    acc_array[fid].write(Legion::DomainPoint::from_point<ndim>(i), x); 
  }
  
  void write(Legion::Point<ndim> i, int fid, a x)
  {
    (acc_array[fid])[i] = x; 
  }
  
  rw_region(context c, Point<ndim> p) : 
  _region<a,ndim>(Rect<ndim>(Point<ndim>::ZEROES(), 
                             p-Point<ndim>::ONES())) 
  {
    std::cout << "set rect to be from " << Point<ndim>::ZEROES() << " to " << this->rect.hi << std::endl; 
    this->is = c.runtime->create_index_space(c.ctx, Legion::Domain::from_rect<ndim>(this->rect));
    Legion::FieldSpace fs = c.runtime->create_field_space(c.ctx);
    Legion::FieldAllocator all = c.runtime->create_field_allocator(c.ctx, fs);
    all.allocate_field(sizeof(a), OnlyField);
    this->l = c.runtime->create_logical_region(c.ctx, this->is, fs);
    this->parent = this->l; 
    nb_fields = 0;
  }; 
  
  rw_region(context& c, IdxSpace<ndim> ispace, FdSpace fspace)
  {
    this->rect = ispace.rect;
    this->is = ispace.idx_space;
    this->l = c.runtime->create_logical_region(c.ctx, this->is, fspace.fd_space);
    this->parent = this->l;
    nb_fields = 0;
  }
  
  void setPhysical(context &c, Legion::PhysicalRegion &pr, Legion::RegionRequirement &rr)
  {
    this->p = pr;
    std::set<Legion::FieldID>::iterator it;
    for (it = rr.privilege_fields.begin(); it != rr.privilege_fields.end(); it++) {
      int field = *(it);
      printf("rr field %d, set acc \n", field);
      acc_array[field] = Legion::FieldAccessor<READ_WRITE, a, ndim>(pr, field);
    }
    
    this->domain = c.runtime->get_index_space_domain(c.ctx, rr.region.get_index_space());
  }
  
  void set_task_field(int fid)
  {
    this->task_fields[nb_fields] = fid;
    nb_fields ++;
  }

  int task_fields[10];
  int nb_fields;
  Legion::FieldAccessor<READ_WRITE, a, ndim> acc_array[10]; 
  const static legion_privilege_mode_t pm = READ_WRITE;
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

template <typename a, size_t ndim, template <typename, size_t> typename t>
inline typename std::enable_if<!std::is_base_of<_region<a,ndim>, t<a,ndim>>::value, int>::type
  bindPhysical(context &c, std::vector<Legion::PhysicalRegion> pr, std::vector<Legion::RegionRequirement> rr, size_t i, t<a,ndim> *r){ return i; }; 

template <typename a, size_t ndim, template <typename, size_t> typename t>
inline typename std::enable_if<std::is_base_of<_region<a,ndim>, t<a,ndim>>::value, int>::type
bindPhysical(context &c, std::vector<Legion::PhysicalRegion> pr, std::vector<Legion::RegionRequirement> rr, size_t i, t<a,ndim> *r){
  r->setPhysical(c, std::ref(pr[i]), std::ref(rr[i]));    
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
inline void registerRR(Legion::TaskLauncher &l, t r){ }; 

template <typename a, size_t ndim, template <typename, size_t> typename t>
inline typename std::enable_if<!std::is_base_of<_region<a,ndim>, t<a,ndim>>::value, void>::type
registerRR(Legion::TaskLauncher &l, t<a, ndim> r){ }; 

template <typename a, size_t ndim, template <typename, size_t> typename t>
inline typename std::enable_if<std::is_base_of<_region<a,ndim>, t<a,ndim>>::value, void>::type
registerRR(Legion::TaskLauncher &l, t<a, ndim> r){
  l.add_region_requirement(r.rr());    
  //printf("registered region\n"); 
};

template<size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
regionArgReqs(Legion::TaskLauncher &l, std::tuple<Tp...> t) {}

template<size_t I = 0, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), void>::type 
regionArgReqs(Legion::TaskLauncher &l, std::tuple<Tp...> t)  {
  registerRR(l, std::get<I>(t));
  regionArgReqs<I+1>(l, t);  
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

public:
  UserTask(const char* name="default")
  {
    id = globalId++; // still not sure about this
  }
  template <typename F, F f>
  void register_task()
  {
    typedef typename function_traits<F>::returnType RT; 
    Legion::ProcessorConstraint pc = Legion::ProcessorConstraint(Legion::Processor::LOC_PROC);
    Legion::TaskVariantRegistrar registrar(id, "name");
    registrar.add_constraint(pc);
    regTask<RT, F, f>::variant(registrar); 
  }
  template <typename F, typename ...Args>
  Future launch_single_task(F f, context c, Args... a)
  {
    typedef typename function_traits<F>::args argtuple;
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
  Future execute_task(F func_ptr, context c, Args... a)
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
  
private:
  UserTask* get_user_task_obj(uintptr_t func_ptr)
  {
    std::map<uintptr_t, UserTask>::iterator it = user_task_map.find(func_ptr);
    if (it != user_task_map.end()) {
      return &(it->second);
    } else {
      printf("can not find task %p\n", func_ptr);
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


template <typename a, size_t ndim> 
void _fill(context _c, w_region<a, ndim> r, a v); 
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
