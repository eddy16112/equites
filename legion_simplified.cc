#include "legion_simplified.h"

namespace LegionSimplified {
  
  // Global id that is indexed for tasks
  static Legion::TaskID globalId = 0;
  
  TaskRuntime runtime;
  
  /////////////////////////////////////////////////////////////
  // FdSpace
  /////////////////////////////////////////////////////////////
  
  //----------------------------------public-------------------------------------
  FdSpace::FdSpace(const context& c) : ctx(c)
  {
    fs = c.runtime->create_field_space(c.ctx);
    allocator = c.runtime->create_field_allocator(c.ctx, fs);
    field_id_vec.clear();
  }

  FdSpace::~FdSpace(void)
  {
    ctx.runtime->destroy_field_space(ctx.ctx, fs);
  }
  
  void FdSpace::add_field(size_t size, field_id_t fid)
  {
    allocator.allocate_field(size,fid);
    field_id_vec.push_back(fid);
  }
  
  /////////////////////////////////////////////////////////////
  // UserTask 
  /////////////////////////////////////////////////////////////
  
  //----------------------------------public-------------------------------------
  UserTask::UserTask(const char* name="default") : task_name(name)
  {
    id = globalId++;
    DEBUG_PRINT((2, "Create new user task name: %s, id %u\n", name, id));
  }
  
  UserTask::~UserTask(void)
  {
  }
  
  /////////////////////////////////////////////////////////////
  // TaskRuntime 
  /////////////////////////////////////////////////////////////
  
  //----------------------------------public-------------------------------------
  TaskRuntime::TaskRuntime(void)
  {
    user_task_map.clear();
    
    inline_map_region_vector.clear();
    
    inline_map_region_t empty_region;
    assert(empty_region.is_empty() == true);
    inline_map_region_vector.push_back(empty_region);
  }
  
  TaskRuntime::~TaskRuntime(void)
  {
  }
  
  UserTask* TaskRuntime::get_user_task_obj(uintptr_t func_ptr)
  {
    std::map<uintptr_t, UserTask>::iterator it = user_task_map.find(func_ptr);
    if (it != user_task_map.end()) {
      return &(it->second);
    } else {
      DEBUG_PRINT((0, "can not find task %p\n", (void*)func_ptr));
      return NULL;
    }
  }
  
  std::vector<inline_map_region_t>::const_iterator TaskRuntime::check_inline_map_conflict(inline_map_region_t &new_region)
  {
    return inline_map_region_vector.cbegin();
  }
  
  void TaskRuntime::add_inline_map(inline_map_region_t &new_region)
  {
    inline_map_region_vector.push_back(new_region);
  }
  
  void TaskRuntime::remove_inline_map(std::vector<inline_map_region_t>::const_iterator it)
  {
    inline_map_region_vector.erase(it);
  }
  
  void debug_printf(int verbose_level, const char *format, ...)
  {
    if (verbose_level > VERBOSE_PRINT) {
      return;
    } else {
      va_list args;
      va_start(args, format);
      vprintf(format, args);
      va_end(args);
      return;
    }
  }
  
}; // namespace LegionSimplified
