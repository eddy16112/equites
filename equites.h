#include <stdio.h>
#include <legion/legion.h>
#include <functional>
#include <iterator> 
#include <algorithm>

namespace equites { 
using namespace std;
using namespace Legion; 
using namespace LegionRuntime::Arrays; 
using namespace LegionRuntime::Accessor; 

// Global id that is indexed for tasks
static TaskID globalId = 0;
// Only have one field currently.  
const static unsigned OnlyField = 0; 

// We define three helper macros. `task` defines a task, while `call` calls a
// task, and `region` creates a rw_region. These aren't strictly necessary, but
// are nice in that they hide the task context from the user and make task
// declaration a slightly easier. 
#define task(type, name, args...) \
  type _##name (context _c, ## args); \
  _task<decltype(&_##name), _##name> name(#name); \
  type _##name (context _c, ## args) 
// call simply hides the context as well
#define call(f, args...) f._call(_c, ## args) 
// Region hides the context call as well
#define region(a, ndim, size) rw_region<a,ndim>(_c, size)

/* A simple wrapper to avoid user-facing casting. */
template <typename a> 
struct future {
  future(Future f) { fut = f; };
  a get(){return fut.get_result<a>();};
  private: 
    Future fut;
};

/* Simple wrapper for all needed context passed to tasks */
struct context {
  const Task *task;
  Context ctx;
  Runtime *runtime;
};

/* regions. */
/* _region is an abstract class */
template <typename a, size_t ndim>
struct _region{
  _region(){}; 
  _region(const Rect<ndim> r) : rect(r) {}; 
  class iterator: public std::iterator <input_iterator_tag, a, Point<ndim>> {
    Point<ndim> pt; 
    GenericPointInRectIterator<ndim> pir; 
    public: 
    explicit iterator(const Rect<ndim> r, Point<ndim> p) : pt(p), pir(GenericPointInRectIterator<ndim>(r)) {} ; 
    iterator& operator++() {pir.step(); pt = pir.p; return *this; }
    iterator operator++(int) {iterator retval = *this; ++(*this); return retval; }
    bool operator==(iterator other) const { return pt == other.pt; }
    bool operator!=(iterator other) const { return !(*this == other); }
    Point<ndim> operator*() const { return pt; }
  }; 

  RegionRequirement rr(){
    RegionRequirement req(this->l, this->pm, this->cp, this->parent); 
    req.add_field(OnlyField); 
    return req; 
  };
  void setPhysical(PhysicalRegion &p){
    this->p = p;
    this->acc = this->p.get_field_accessor(OnlyField).template typeify<a>();
  }
  RegionAccessor<AccessorType::Generic, a> acc; 
  PhysicalRegion p;
  LogicalRegion l; 
  LogicalRegion parent; 
  const Rect<ndim> rect; 
  const static legion_privilege_mode_t pm = NO_ACCESS;  
  const static legion_coherence_property_t cp = EXCLUSIVE; 
  iterator begin() { return iterator(rect, rect.lo); }
  iterator end() { return iterator(rect, rect.hi); }
};

// read only region
template <typename a, size_t ndim>
struct r_region : virtual _region<a, ndim> {
  r_region(){}; 
  RegionRequirement rr(){
    RegionRequirement req(this->l, this->pm, this->cp, this->parent); 
    req.add_field(OnlyField); 
    return req; 
  };
  a read(Point<ndim> i){
    return this->acc.read(DomainPoint::from_point<ndim>(i));
  };
  const static legion_privilege_mode_t pm = READ_ONLY;  
};

// write only region
template <typename a, size_t ndim>
struct w_region : virtual _region<a, ndim> {
  w_region(){}; 
  RegionRequirement rr(){
    RegionRequirement req(this->l, this->pm, this->cp, this->parent); 
    req.add_field(OnlyField); 
    return req; 
  };
  void write(Point<ndim> i, a x) {
    this->acc.write(DomainPoint::from_point<ndim>(i), x); 
  }
  const static legion_privilege_mode_t pm = WRITE_ONLY;  
};

// read-write region
template <typename a, size_t ndim>
struct rw_region : virtual r_region<a, ndim>, virtual w_region<a, ndim> {
  RegionRequirement rr(){
    RegionRequirement req(this->l, this->pm, this->cp, this->parent); 
    req.add_field(OnlyField); 
    return req; 
  };
  rw_region(context c, Point<ndim> p) : _region<a,ndim>(Rect<ndim>(Point<ndim>::ZEROES(), p)) {
    //cout << "set rect to be from " << Point<ndim>::ZEROES() << " to " << this->rect.hi << endl; 
    IndexSpace is = c.runtime->create_index_space(c.ctx, Domain::from_rect<ndim>(this->rect));
    FieldSpace fs = c.runtime->create_field_space(c.ctx);
    FieldAllocator all = c.runtime->create_field_allocator(c.ctx, fs);
    all.allocate_field(sizeof(a), OnlyField);
    this->l = c.runtime->create_logical_region(c.ctx, is, fs);
    this->parent = this->l; 
  }; 
  const static legion_privilege_mode_t pm = READ_WRITE;
};

// Helper to apply a task to tuple 
template <class T, class F, size_t... Is>
inline auto apply_impl(T t, F f, index_sequence<Is...>){
  return f(get<Is>(t)...);
}

template <class F, class T>
inline auto apply(F f, T t){
  return apply_impl(t, f, make_index_sequence<tuple_size<T>{}>{});
}

// Helper to extract return type and argument types from function type  
template<typename T> struct function_traits {};
template<typename R, typename ...Args>
struct function_traits<R (*) (context c, Args...)> {
  static constexpr size_t nargs = sizeof...(Args);
  typedef R returnType;
  typedef tuple<Args...> args;
};

template<typename R, typename ...Args>
struct function_traits<R (*) (Args...)> {
  static constexpr size_t nargs = sizeof...(Args);
  typedef R returnType;
  typedef tuple<Args...> args;
};

// Specialization on regions
template <typename t> 
inline int bindPhysical(vector<PhysicalRegion> v, size_t i, t x) { return i; }; 

template <typename a, size_t ndim, template <typename, size_t> class t>
inline typename enable_if<!is_base_of<_region<a,ndim>, t<a,ndim>>::value, int>::type
  bindPhysical(vector<PhysicalRegion> v, size_t i, t<a,ndim> *r){ return i; }; 

template <typename a, size_t ndim, template <typename, size_t> class t>
inline typename enable_if<is_base_of<_region<a,ndim>, t<a,ndim>>::value, int>::type
bindPhysical(vector<PhysicalRegion> v, size_t i, t<a,ndim> *r){
  // printf("mapped region %d", i); 
  r->setPhysical(ref(v[i]));    
  return i+1; 
};

// Template size_t
template<size_t I = 0, typename... Tp>
inline typename enable_if<I == sizeof...(Tp), void>::type
bindPs(vector<PhysicalRegion> v, size_t i, tuple<Tp...> *) { }

template<size_t I = 0, typename... Tp>
inline typename enable_if<I < sizeof...(Tp), void>::type 
bindPs(vector<PhysicalRegion> v, size_t i, tuple<Tp...> *t){
  bindPs<I+1>(v, bindPhysical(v, i, &get<I>(*t)), t); 
};

// Builds a legion task out of an arbitrary function that takes a context as a
// first argument
template <typename F, F f>
inline typename function_traits<F>::returnType  
mkLegionTask(const Task *task, const vector<PhysicalRegion>& rs, Context ctx, Runtime* rt) {
  typedef typename function_traits<F>::args argtuple;
  argtuple at = *(argtuple*) task->args;
  context c = { task, ctx, rt };
  size_t i = 0; 
  bindPs(rs, i, &at); 
  return apply(f, tuple_cat(make_tuple(c), at));
}

// Process region requirements for function calls
template <typename t> 
inline void registerRR(TaskLauncher &l, t r){ }; 

template <typename a, size_t ndim, template <typename, size_t> class t>
inline typename enable_if<!is_base_of<_region<a,ndim>, t<a,ndim>>::value, void>::type
registerRR(TaskLauncher &l, t<a, ndim> r){ }; 

template <typename a, size_t ndim, template <typename, size_t> class t>
inline typename enable_if<is_base_of<_region<a,ndim>, t<a,ndim>>::value, void>::type
registerRR(TaskLauncher &l, t<a, ndim> r){
  l.add_region_requirement(r.rr());    
  //printf("registered region\n"); 
};

template<size_t I = 0, typename... Tp>
inline typename enable_if<I == sizeof...(Tp), void>::type
regionArgReqs(TaskLauncher &l, tuple<Tp...> t) {}

template<size_t I = 0, typename... Tp>
inline typename enable_if<I < sizeof...(Tp), void>::type 
regionArgReqs(TaskLauncher &l, tuple<Tp...> t)  {
  registerRR(l, get<I>(t));
  regionArgReqs<I+1>(l, t);  
}

template <typename T, typename F, F f> 
struct regTask {
  static void variant(TaskVariantRegistrar r){
    Runtime::preregister_task_variant<T, mkLegionTask<F, f>>(r);
  }
};

template <typename F, F f>
struct regTask<void, F, f> {
  static void variant(TaskVariantRegistrar r){
    Runtime::preregister_task_variant<mkLegionTask<F, f>>(r);
  }
};

template <typename F, F f>
class _task {
  public:
  typedef typename function_traits<F>::returnType RT; 
  typedef typename function_traits<F>::args argtuple;
  TaskID id;
  _task(const char* name="default", 
        ProcessorConstraint pc = 
          ProcessorConstraint(Processor::LOC_PROC)){
    id = globalId++; // still not sure about this 
    TaskVariantRegistrar registrar(id, name);
    registrar.add_constraint(pc);
    regTask<RT, F, f>::variant(registrar); 
  }
  template <typename ...Args>
  future<RT> _call(context c, Args... a){
    argtuple p = make_tuple(a...);
    TaskLauncher l(id, TaskArgument(&p, sizeof(p))); 
    regionArgReqs(l, p);  
    return future<RT>(c.runtime->execute_task(c.ctx, l));
  }
};

template <void (*f) (context)>
int start(_task<void (*) (context), f> t, int argc, char** argv){ 
  Runtime::set_top_level_task_id(t.id);
  return Runtime::start(argc, argv);
};

template <void (*f) (context, int, char**)>
int start(_task<void (*) (context, int, char**), f> t, int argc, char** argv){ 
  Runtime::set_top_level_task_id(t.id);
  return Runtime::start(argc, argv);
};

} /* namespace Equites */
