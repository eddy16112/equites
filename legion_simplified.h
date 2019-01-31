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

#define VERBOSE_PRINT     6

#define DEBUG_OUTPUT      1

namespace LegionSimplified { 
  
#ifdef DEBUG_OUTPUT
# define DEBUG_PRINT(x) debug_printf x
#else
# define DEBUG_PRINT(x)
#endif
  
  enum partition_type
  {
    equal,
    restriction,
  };
  
  struct inline_map_region_t {
  public:
    Legion::RegionRequirement rr;
    Legion::PhysicalRegion pr;
  public:
    bool inline is_empty()
    {
      if (rr.privilege_fields.size() == 0) {
        return true;
      } else {
        return false;
      }
    }
  };

  template <size_t DIM>  using Point = Legion::Point<DIM>;  
  template <size_t DIM>  using Rect = Legion::Rect<DIM>;  

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
    const std::vector<field_id_t> &field_id_vec;
    Legion::LogicalRegion lr; 
    Legion::LogicalRegion lr_parent;
  
  public:
    Region(IdxSpace<DIM> &ispace, FdSpace &fspace);
    
    Region(const context &c, const std::vector<field_id_t> &field_id_vec, Legion::LogicalRegion &lr, Legion::LogicalRegion &lr_parent);
  
    ~Region(void);
  };

  /**
   * \class Partition
   * A class for representing IndexPartition and LogicalPartition
   */
  template <size_t DIM>
  class Partition {
  public:
    const context &ctx;
    const Region<DIM> &region_parent;
    Legion::IndexPartition ip;
    Legion::LogicalPartition lp;
  
  public:
    Partition(int p_type, Region<DIM> &r, IdxSpace<DIM> &ispace);
  
    Partition(int p_type, Region<DIM> &r, IdxSpace<DIM> &ispace, Legion::DomainTransform &dt, Rect<DIM> &rect);
  
    ~Partition(void);
    
    Region<DIM> get_subregion_by_color(int color);
    
    Region<DIM> operator[] (int color);
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
      Future fu(fm.get_future(point)); 
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

  /**
   * \class FutureMap
   * A class for representing ArgumentMap
   */
  class ArgMap {
  public:
    Legion::ArgumentMap arg_map;
  public:
    ArgMap(void)
    {
    }
    template<typename T, size_t DIM>
    void set_point(const Point<DIM> &point, T val)
    {
      arg_map.set_point(point, Legion::TaskArgument(&val, sizeof(T)));
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
    Legion::LogicalRegion lr;  // mutually exclusive with lp
    Legion::LogicalPartition lp; // mutually exclusive with lr
    Legion::LogicalRegion lr_parent;
    int is_mapped;
    Legion::Domain domain;
    Legion::PhysicalRegion physical_region;
    std::vector<field_id_t> field_id_vector;
    std::map<field_id_t, unsigned char*> accessor_map;
    
  public:
    BaseRegionImpl(void);
    
    ~BaseRegionImpl(void);
    
    void init_accessor_map(void);
  };

  /**
   * \class Base_Region
   * A class for representing 
   */
  template <size_t DIM>
  class Base_Region {
  public:
    const context *ctx;
  
    //BaseRegionImpl<DIM> *base_region_impl;
    std::shared_ptr<BaseRegionImpl<DIM>> base_region_impl;
  
    legion_privilege_mode_t pm; 
    const static legion_coherence_property_t cp = EXCLUSIVE; 
  
  public:
    Base_Region(void);
  
    Base_Region(const Base_Region &rhs);
  
    Base_Region(Region<DIM> &r, std::vector<field_id_t> &task_field_id_vec);
  
    Base_Region(Region<DIM> &r);
  
    Base_Region(Partition<DIM> &par, std::vector<field_id_t> &task_field_id_vec);
  
    Base_Region(Partition<DIM> &par);
  
    ~Base_Region(void);
  
    Base_Region & operator=(const Base_Region &rhs);
  
    Legion::RegionRequirement set_region_requirement_single(void);
  
    Legion::RegionRequirement set_region_requirement_index(void);
  
    void map_physical_region(context &c, Legion::PhysicalRegion &pr, Legion::RegionRequirement &rr);
  
    void map_physical_region_inline_with_auto_unmap();
    
    void map_physical_region_inline(void);
  
    void unmap_physical_region_inline(void);
  
    void cleanup_reference(void);
    
    void if_mapped(void);
    
    Region<DIM> get_region(void);
  
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
        if (is_valid == true && other.is_valid == true) {
          const Legion::Point<DIM> my_pt = operator*();
          const Legion::Point<DIM> other_pt = other.operator*();
          return (my_pt != other_pt);
        } else {
          if (is_valid == other.is_valid) {
            return false;
          } else {
            return true;
          }
        }
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
    void init_parameters(void);
  
    void check_empty(void);
    
  };


  /**
   * \class RO_Region
   * A class for representing RO region
   */
  template <size_t DIM>
  class RO_Region : public Base_Region<DIM> {
  public:
    RO_Region(void);

    RO_Region(Region<DIM> &r, std::vector<field_id_t> &task_field_id_vec);
  
    RO_Region(Region<DIM> &r);
  
    RO_Region(Partition<DIM> &par, std::vector<field_id_t> &task_field_id_vec);
  
    RO_Region(Partition<DIM> &par);
  
    ~RO_Region(void);

    template< typename a>
    a read(int fid, Legion::Point<DIM> i);
  
    template< typename a>
    a read(Legion::Point<DIM> i);
     
    void init_ro_parameters(void);
  
  private:
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

    WD_Region(Region<DIM> &r, std::vector<field_id_t> &task_field_id_vec);
  
    WD_Region(Region<DIM> &r);
  
    WD_Region(Partition<DIM> &par, std::vector<field_id_t> &task_field_id_vec);
  
    WD_Region(Partition<DIM> &par);
  
    ~WD_Region(void);
    
    template< typename a>
    void write(int fid, Legion::Point<DIM> i, a x);
  
    template< typename a>
    void write(Legion::Point<DIM> i, a x);
   
    void init_wd_parameters(void);
  
  private: 
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

    RW_Region(Region<DIM> &r, std::vector<field_id_t> &task_field_id_vec);
  
    RW_Region(Region<DIM> &r);
  
    RW_Region(Partition<DIM> &par, std::vector<field_id_t> &task_field_id_vec);
    
    RW_Region(Partition<DIM> &par);
  
    ~RW_Region(void);
    
    template< typename a>
    a read(int fid, Legion::Point<DIM> i);
  
    template< typename a>
    a read(Legion::Point<DIM> i);
  
    template< typename a>
    void write(int fid, Legion::Point<DIM> i, a x);
  
    template< typename a>
    void write(Legion::Point<DIM> i, a x);
 
    void init_rw_parameters(void);

  private:   
    template< typename a>
    Legion::FieldAccessor<READ_WRITE, a, DIM>* get_accessor_by_fid(field_id_t fid);
  
    template< typename a>
    Legion::FieldAccessor<READ_WRITE, a, DIM>* get_default_accessor(void);
  };
  
  template <size_t DIM>
  class RO_Partition : public RO_Region<DIM> {
  public:      
    RO_Partition(void);
  
    RO_Partition(Partition<DIM> &par, std::vector<field_id_t> &task_field_id_vec);
  
    RO_Partition(Partition<DIM> &par);
  
    ~RO_Partition(void);
  };
  
  template <size_t DIM>
  class WD_Partition : public WD_Region<DIM> {
  public:      
    WD_Partition(void);
  
    WD_Partition(Partition<DIM> &par, std::vector<field_id_t> &task_field_id_vec);
  
    WD_Partition(Partition<DIM> &par);
  
    ~WD_Partition(void);
  };
  
  template <size_t DIM>
  class RW_Partition : public RW_Region<DIM> {
  public:      
    RW_Partition(void);
  
    RW_Partition(Partition<DIM> &par, std::vector<field_id_t> &task_field_id_vec);
  
    RW_Partition(Partition<DIM> &par);
  
    ~RW_Partition(void);
  };

//-----------------------------------------------------------------------------
// Helper to extract return type and argument types from function type  
  template<typename T> struct function_traits {};
  template<typename R, typename ...Args>
  struct function_traits<R (*) (context c, Args...)> 
  {
    static constexpr size_t nargs = sizeof...(Args);
    typedef R returnType;
    typedef std::tuple<Args...> args;
  };

  template<typename R, typename ...Args>
  struct function_traits<R (*) (Args...)> 
  {
    static constexpr size_t nargs = sizeof...(Args);
    typedef R returnType;
    typedef std::tuple<Args...> args;
  };

//-----------------------------------------------------------------------------
// map physical region before going to task function
  template <typename t> 
  inline int bind_physical_region(context &c, std::vector<Legion::PhysicalRegion> pr, 
    std::vector<Legion::RegionRequirement> rr, size_t i, t x) 
  { 
    return i; 
  } 

  template <size_t DIM, template <size_t> typename t>
  inline typename std::enable_if<!std::is_base_of<Base_Region<DIM>, t<DIM>>::value, int>::type
  bind_physical_region(context &c, std::vector<Legion::PhysicalRegion> pr, 
    std::vector<Legion::RegionRequirement> rr, size_t i, t<DIM> *base_region)
  { 
    return i; 
  } 

  //TODO std::ref
  template <size_t DIM, template <size_t> typename t>
  inline typename std::enable_if<std::is_base_of<Base_Region<DIM>, t<DIM>>::value, int>::type
  bind_physical_region(context &c, std::vector<Legion::PhysicalRegion> pr, 
    std::vector<Legion::RegionRequirement> rr, size_t i, t<DIM> *base_region)
  {
    base_region->map_physical_region(c, std::ref(pr[i]), std::ref(rr[i]));    
    return i+1; 
  }

  // Template size_t
  template<size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
  bind_physical_region_tuple_walker(context &c, const std::vector<Legion::PhysicalRegion> &pr, 
    const std::vector<Legion::RegionRequirement> &rr, size_t i, std::tuple<Tp...> *) {}

  template<size_t I = 0, typename... Tp>
  inline typename std::enable_if<I < sizeof...(Tp), void>::type 
  bind_physical_region_tuple_walker(context &c, const std::vector<Legion::PhysicalRegion> &pr, 
    const std::vector<Legion::RegionRequirement> &rr, size_t i, std::tuple<Tp...> *t)
  {
    bind_physical_region_tuple_walker<I+1>(c, pr, rr, 
      bind_physical_region(c, pr, rr, i, &std::get<I>(*t)), t); 
  }

//-----------------------------------------------------------------------------
// Helper to apply a task to tuple 
  template <typename T, typename F, size_t... Is>
  inline auto apply_task_impl(T t, F f, std::index_sequence<Is...>)
  {
    return f(std::get<Is>(t)...);
  }

  template <typename F, typename T>
  inline auto apply_task(F f, T t)
  {
    return apply_task_impl(t, f, std::make_index_sequence<std::tuple_size<T>{}>{});
  }

//-----------------------------------------------------------------------------
// builds a legion task out of an arbitrary function that takes a context as a
// first argument
  template <typename F, F f>
  inline typename function_traits<F>::returnType  
  make_legion_task(const Legion::Task *task, const std::vector<Legion::PhysicalRegion>& pr, 
    Legion::Context ctx, Legion::Runtime* rt) 
  {
    typedef typename function_traits<F>::args argtuple;
    argtuple at = *(argtuple*) task->args;
    context c = { task, ctx, rt };
    size_t i = 0; 
    bind_physical_region_tuple_walker(c, pr, task->regions, i, &at); 
    return apply_task(f, std::tuple_cat(std::make_tuple(c), at));
  }

//----------------------------------------------------------------------------------
// clean up the shared ptr of base_region before launcher tasks
  template <typename t> 
  inline void base_region_cleanup_shared_ptr(t &base_region) {} 

  template <size_t DIM, template <size_t> typename t>
  inline typename std::enable_if<!std::is_base_of<Base_Region<DIM>, t<DIM>>::value, void>::type
  base_region_cleanup_shared_ptr(t<DIM> &base_region) {} 

  template <size_t DIM, template <size_t> typename t>
  inline typename std::enable_if<std::is_base_of<Base_Region<DIM>, t<DIM>>::value, void>::type
  base_region_cleanup_shared_ptr(t<DIM> &base_region)
  {
    base_region.cleanup_reference();  
    //printf("registered region\n"); 
  }

  template<size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
  base_region_cleanup_shared_ptr_tuple_walker(std::tuple<Tp...> &t) {}

  template<size_t I = 0, typename... Tp>
  inline typename std::enable_if<I < sizeof...(Tp), void>::type 
  base_region_cleanup_shared_ptr_tuple_walker(std::tuple<Tp...> &t)
  {
    base_region_cleanup_shared_ptr(std::get<I>(t));
    base_region_cleanup_shared_ptr_tuple_walker<I+1>(t);  
  }

//----------------------------------------------------------------------------------
// add region requirement for task launcher
  template <typename t> 
  inline void task_launcher_add_region_requirement(Legion::TaskLauncher &task_launcher, t &base_region) {} 

  template <size_t DIM, template <size_t> typename t>
  inline typename std::enable_if<!std::is_base_of<Base_Region<DIM>, t<DIM>>::value, void>::type
  task_launcher_add_region_requirement(Legion::TaskLauncher &task_launcher, t<DIM> &base_region) {} 

  template <size_t DIM, template <size_t> typename t>
  inline typename std::enable_if<std::is_base_of<Base_Region<DIM>, t<DIM>>::value, void>::type
  task_launcher_add_region_requirement(Legion::TaskLauncher &task_launcher, t<DIM> &base_region)
  {
    task_launcher.add_region_requirement(base_region.set_region_requirement_single());    
    //printf("registered region\n"); 
  }

  template<size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
  task_launcher_region_requirement_tuple_walker(Legion::TaskLauncher &l, std::tuple<Tp...> &t) {}

  template<size_t I = 0, typename... Tp>
  inline typename std::enable_if<I < sizeof...(Tp), void>::type 
  task_launcher_region_requirement_tuple_walker(Legion::TaskLauncher &l, std::tuple<Tp...> &t) 
  {
    task_launcher_add_region_requirement(l, std::get<I>(t));
    task_launcher_region_requirement_tuple_walker<I+1>(l, t);  
  }

//----------------------------------------------------------------------------------
// add region requirement for index launcher
  template <typename t> 
  inline void index_launcher_add_region_requirement(Legion::IndexLauncher &index_launcher, t &base_region) {} 

  template <size_t DIM, template <size_t> typename t>
  inline typename std::enable_if<!std::is_base_of<Base_Region<DIM>, t<DIM>>::value, void>::type
  index_launcher_add_region_requirement(Legion::IndexLauncher &index_launcher, t<DIM> &base_region) {} 

  template <size_t DIM, template <size_t> typename t>
  inline typename std::enable_if<std::is_base_of<Base_Region<DIM>, t<DIM>>::value, void>::type
  index_launcher_add_region_requirement(Legion::IndexLauncher &index_launcher, t<DIM> &base_region)
  {
    index_launcher.add_region_requirement(base_region.set_region_requirement_index());    
    //printf("registered region\n"); 
  }

  template<size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
  index_launcher_region_requirement_tuple_walker(Legion::IndexLauncher &index_launcher, std::tuple<Tp...> &t) {}

  template<size_t I = 0, typename... Tp>
  inline typename std::enable_if<I < sizeof...(Tp), void>::type 
  index_launcher_region_requirement_tuple_walker(Legion::IndexLauncher &index_launcher, std::tuple<Tp...> &t)
  {
    index_launcher_add_region_requirement(index_launcher, std::get<I>(t));
    index_launcher_region_requirement_tuple_walker<I+1>(index_launcher, t);  
  }

//----------------------------------------------------------------------------------
// legion register task constraint and variant
  template <typename T, typename F, F f> 
  struct TaskRegistration {
    static void variant(const Legion::TaskVariantRegistrar &r, const char *task_name = NULL){
      Legion::Runtime::preregister_task_variant<T, make_legion_task<F, f>>(r, task_name);
    }
  };

  template <typename F, F f>
  struct TaskRegistration<void, F, f> {
    static void variant(const Legion::TaskVariantRegistrar &r, const char *task_name = NULL){
      Legion::Runtime::preregister_task_variant<make_legion_task<F, f>>(r, task_name);
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
    void register_task(bool leaf);
    
    // launch single task
    template <typename F, typename ...Args>
    Future launch_single_task(F f, context &c, Args... a);
    
    // launch index task  
    template <size_t DIM, typename F, typename ...Args>
    FutureMap launch_index_task(F f, context &c, IdxSpace<DIM> &ispace, Args... a);
  
    // launch index task with argmap
    template <size_t DIM, typename F, typename ...Args>
    FutureMap launch_index_task(F f, context &c, IdxSpace<DIM> &ispace, ArgMap argmap, Args... a);
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

  /**
   * \class TaskRuntime
   * A class for runtime.
   */
  class TaskRuntime
  {
  private:
    // a map to query tasks by function ptr
    std::map<uintptr_t, UserTask> user_task_map;
    std::vector<inline_map_region_t> inline_map_region_vector;
  
  public:
    TaskRuntime(void);
    
    ~TaskRuntime(void);
  
    // register tasks 
    template <typename F, F func_ptr>
    void register_task(const char* name);
    
    // register tasks 
    template <typename F, F func_ptr>
    void register_task(const char* name, bool leaf);
  
    // start runtime
    template <typename F>
    int start(F func_ptr, int argc, char** argv);
  
    // task launcher
    template <typename F, typename ...Args>
    Future execute_task(F func_ptr, context &c, Args... a);
  
    // index task launcher
    template <size_t DIM, typename F, typename ...Args>
    FutureMap execute_task(F func_ptr, context &c, IdxSpace<DIM> &is, Args... a);
  
    // index task launcher with argmap
    template <size_t DIM, typename F, typename ...Args>
    FutureMap execute_task(F func_ptr, context &c, IdxSpace<DIM> &is, ArgMap argmap, Args... a);
    
    std::vector<inline_map_region_t>::const_iterator check_inline_map_conflict(inline_map_region_t &new_region);
    
    void add_inline_map(inline_map_region_t &new_region);
    
    void remove_inline_map(std::vector<inline_map_region_t>::const_iterator it);
  
  private:
    // query task by task function ptr
    UserTask* get_user_task_obj(uintptr_t func_ptr);  
  };

  extern TaskRuntime runtime;
  
  void debug_printf(int verbose_level, const char *format, ...);

}; // namespace LegionSimplified

#include "legion_simplified.inl"

#endif // _LEGION_SIMPLIFIED_H
