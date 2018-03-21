#include <stdio.h>
#include <legion.h>
#include <functional>
#include <iterator> 
#include <algorithm>
#include <array>

namespace equites { 

template <size_t ndim> 
using Point = LegionRuntime::Arrays::Point<ndim>;  
using LegionRuntime::Arrays::make_point;

// Global id that is indexed for tasks
static Legion::TaskID globalId = 0;

// Only have one field currently.  
const static size_t OnlyField = 0; 

// would be nice to make this generic for n dimensions
/*
template <size_t ndim>
LegionRuntime::Arrays::Point<ndim> Point(array<long long,ndim> a)
  { return LegionRuntime::Arrays::Point<ndim>((long long[ndim]) a.data); }
*/

// We define three helper macros. `task` defines a task, while `call` calls a
// task, and `region` creates a rw_region. These aren't strictly necessary, but
// are nice in that they hide the task context from the user and make task
// declaration a slightly easier. 
#define task(type, name, args...) \
  type _##name (context _c, ## args); \
  _task<decltype(&_##name), _##name> name(#name); \
  type _##name (context _c, ## args) 
// call and icall simply hide the context as well
#define call(f, args...) f._call(_c, ## args) 
#define icall(f, args...) f._icall(_c, ## args)
// Region hides the context call as well
#define region(a, ndim, size) rw_region<a,ndim>(_c, size)

/* A simple wrapper to avoid user-facing casting. */
template <typename a> 
struct Future {
  Future(Legion::Future f) { fut = f; };
  a get(){return fut.get_result<a>();};
  private: 
    Legion::Future fut;
};

template <typename a> 
struct FutureMap {
  FutureMap(Legion::FutureMap f) { fut = f; };
  inline Future<a> operator[](const Legion::DomainPoint &point)
    { return Future<a>(fut.get_future(point)); }
  private: 
    Legion::FutureMap fut;
  void wait() {fut.wait_all_results(); } // could return a container of `a`s, e.g. vector<a>
};

/* Simple wrapper for all needed context passed to tasks */
struct context {
  const Legion::Task *task;
  Legion::Context ctx;
  Legion::Runtime *runtime;
};

template <size_t ndim>
static LegionRuntime::Arrays::Point<ndim> END(void) {
  LegionRuntime::Arrays::Point<ndim> z; for(size_t i=0; i < ndim; i++) z.x[i] = -1; return z; 
}

/* regions. */
/* _region is an abstract class */
template <typename a, size_t ndim>
struct _region{
  _region(){}; 
  _region(const LegionRuntime::Arrays::Rect<ndim> r) : rect(r) {}; 
  class iterator: public std::iterator <std::input_iterator_tag, a, LegionRuntime::Arrays::Point<ndim>> {
    public: 
    LegionRuntime::Arrays::Point<ndim> pt; 
    LegionRuntime::Arrays::Rect<ndim> r; 
    explicit iterator(const LegionRuntime::Arrays::Rect<ndim> rect, LegionRuntime::Arrays::Point<ndim> p) : pt(p), r(rect) {} ; 
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
    LegionRuntime::Arrays::Point<ndim> operator*() const { return pt; }
  }; 

  Legion::RegionRequirement rr(){
    Legion::RegionRequirement req(this->l, this->pm, this->cp, this->parent); 
    req.add_field(OnlyField); 
    return req; 
  };

  void setPhysical(Legion::PhysicalRegion &p){
    this->p = p;
    this->acc = Legion::FieldAccessor<NO_ACCESS, a, ndim>(p, OnlyField);
  }

  Legion::FieldAccessor<NO_ACCESS, a, ndim> acc; 
  Legion::PhysicalRegion p;
  Legion::LogicalRegion l; 
  Legion::IndexSpace is; 
  Legion::LogicalRegion parent; 
  const LegionRuntime::Arrays::Rect<ndim> rect; 
  const static legion_privilege_mode_t pm = NO_ACCESS;  
  const static legion_coherence_property_t cp = EXCLUSIVE; 
  iterator begin() { return iterator(rect, rect.lo); }
  iterator end() { return iterator(rect, END<ndim>()); }
};

// read only region
template <typename a, size_t ndim>
struct r_region : virtual _region<a, ndim> {
  r_region(){}; 
  Legion::RegionRequirement rr(){
    Legion::RegionRequirement req(this->l, this->pm, this->cp, this->parent); 
    req.add_field(OnlyField); 
    return req; 
  };
  a read(LegionRuntime::Arrays::Point<ndim> i){
    return this->acc.read(Legion::DomainPoint::from_point<ndim>(i));
  };
  void setPhysical(Legion::PhysicalRegion &p){
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
  void write(LegionRuntime::Arrays::Point<ndim> i, a x){
    this->acc.write(Legion::DomainPoint::from_point<ndim>(i), x); 
  }
  void setPhysical(Legion::PhysicalRegion &p){
    this->p = p;
    this->acc = Legion::FieldAccessor<WRITE_ONLY, a, ndim>(p, OnlyField);
  }

  Legion::FieldAccessor<WRITE_ONLY, a, ndim> acc; 
  const static legion_privilege_mode_t pm = WRITE_ONLY;  
};

// read-write region
template <typename a, size_t ndim>
struct rw_region : virtual r_region<a, ndim>, virtual w_region<a, ndim> {
  Legion::RegionRequirement rr(){
    Legion::RegionRequirement req(this->l, this->pm, this->cp, this->parent); 
    req.add_field(OnlyField); 
    return req; 
  };
  rw_region(context c, LegionRuntime::Arrays::Point<ndim> p) : _region<a,ndim>(LegionRuntime::Arrays::Rect<ndim>(LegionRuntime::Arrays::Point<ndim>::ZEROES(), p-LegionRuntime::Arrays::Point<ndim>::ONES())) {
    //cout << "set rect to be from " << Point<ndim>::ZEROES() << " to " << this->rect.hi << endl; 
    this->is = c.runtime->create_index_space(c.ctx, Legion::Domain::from_rect<ndim>(this->rect));
    Legion::FieldSpace fs = c.runtime->create_field_space(c.ctx);
    Legion::FieldAllocator all = c.runtime->create_field_allocator(c.ctx, fs);
    all.allocate_field(sizeof(a), OnlyField);
    this->l = c.runtime->create_logical_region(c.ctx, this->is, fs);
    this->parent = this->l; 
  }; 
  void setPhysical(Legion::PhysicalRegion &p){
    this->p = p;
    this->acc = Legion::FieldAccessor<READ_WRITE, a, ndim>(p, OnlyField);
  }

  Legion::FieldAccessor<READ_WRITE, a, ndim> acc; 
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
inline int bindPhysical(std::vector<Legion::PhysicalRegion> v, size_t i, t x) { return i; }; 

template <typename a, size_t ndim, template <typename, size_t> typename t>
inline typename std::enable_if<!std::is_base_of<_region<a,ndim>, t<a,ndim>>::value, int>::type
  bindPhysical(std::vector<Legion::PhysicalRegion> v, size_t i, t<a,ndim> *r){ return i; }; 

template <typename a, size_t ndim, template <typename, size_t> typename t>
inline typename std::enable_if<std::is_base_of<_region<a,ndim>, t<a,ndim>>::value, int>::type
bindPhysical(std::vector<Legion::PhysicalRegion> v, size_t i, t<a,ndim> *r){
  r->setPhysical(std::ref(v[i]));    
  return i+1; 
};

// Template size_t
template<size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
bindPs(std::vector<Legion::PhysicalRegion> v, size_t i, std::tuple<Tp...> *) { }

template<size_t I = 0, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), void>::type 
bindPs(std::vector<Legion::PhysicalRegion> v, size_t i, std::tuple<Tp...> *t){
  bindPs<I+1>(v, bindPhysical(v, i, &std::get<I>(*t)), t); 
};

// Builds a legion task out of an arbitrary function that takes a context as a
// first argument
template <typename F, F f>
inline typename function_traits<F>::returnType  
mkLegionTask(const Legion::Task *task, const std::vector<Legion::PhysicalRegion>& rs, Legion::Context ctx, Legion::Runtime* rt) {
  typedef typename function_traits<F>::args argtuple;
  argtuple at = *(argtuple*) task->args;
  context c = { task, ctx, rt };
  size_t i = 0; 
  bindPs(rs, i, &at); 
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

template <typename F, F f>
class _task {
  public:
  typedef typename function_traits<F>::returnType RT; 
  typedef typename function_traits<F>::args argtuple;
  Legion::TaskID id;
  _task(const char* name="default", 
        Legion::ProcessorConstraint pc = 
          Legion::ProcessorConstraint(Legion::Processor::LOC_PROC)){
    id = globalId++; // still not sure about this 
    Legion::TaskVariantRegistrar registrar(id, name);
    registrar.add_constraint(pc);
    regTask<RT, F, f>::variant(registrar); 
  }
  template <typename ...Args>
  Future<RT> _call(context c, Args... a){
    argtuple p = std::make_tuple(a...);
    Legion::TaskLauncher l(id, Legion::TaskArgument(&p, sizeof(p))); 
    regionArgReqs(l, p);  
    return Future<RT>(c.runtime->execute_task(c.ctx, l));
  }
  /*
  template <size_t ndim, typename ...Args>
  FutureMap<RT> _icall(context c, LegionRuntime::Arrays::Point<ndim> pt, Args... a){
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

enum PartitionType {
  equal
}; 
template <typename a, size_t ndim, typename t>
class Partition{
  class iterator: public std::iterator <std::input_iterator_tag, t, LegionRuntime::Arrays::Point<ndim>> {
    public: 
    LegionRuntime::Arrays::Point<ndim> pt; 
    LegionRuntime::Arrays::Rect<ndim> r; 
    explicit iterator(const LegionRuntime::Arrays::Rect<ndim> rect, LegionRuntime::Arrays::Point<ndim> p) : pt(p), r(rect) {} ; 
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
    LegionRuntime::Arrays::Point<ndim> operator*() const { return pt; }
  }; 
  public:
    Partition(context c, PartitionType pt, t parent, LegionRuntime::Arrays::Point<ndim> p)
      : pt(pt), parent(parent) {
      LegionRuntime::Arrays::Rect<ndim> r(LegionRuntime::Arrays::Point<ndim>::ZEROES(), 
                                          p-LegionRuntime::Arrays::Point<ndim>::ONES()); 
      is = c.runtime->create_index_space(c.ctx, r); 
      lp = c.runtime->get_logical_partition(r, is); 
    }
    Legion::IndexSpace is; 
    Legion::LogicalPartition lp;
    Legion::IndexPartition ip;  
    PartitionType pt; 
    t parent; 
}; 

template <void (*f) (context)>
int start(_task<void (*) (context), f> t, int argc, char** argv){ 
  Legion::Runtime::set_top_level_task_id(t.id);
  return Legion::Runtime::start(argc, argv);
};

template <void (*f) (context, int, char**)>
int start(_task<void (*) (context, int, char**), f> t, int argc, char** argv){ 
  Legion::Runtime::set_top_level_task_id(t.id);
  return Legion::Runtime::start(argc, argv);
};


template <typename a, size_t ndim> 
void _fill(context _c, w_region<a, ndim> r, a v); 
template <typename a, size_t ndim> 
_task<decltype(&_fill<a,ndim>), _fill<a,ndim>> fill("fill"); 
template <typename a, size_t ndim>
void _fill(context _c, w_region<a, ndim> r, a v){
  for(auto i : r) r.write(i, v);
}

} /* namespace Equites */
