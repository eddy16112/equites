#include "legion_simplified.h"

namespace LegionSimplified {
  TaskRuntime runtime;

  /////////////////////////////////////////////////////////////
  // Fdpace
  /////////////////////////////////////////////////////////////
  
  FdSpace::FdSpace(const context& c) : ctx(c)
  {
    fs = c.runtime->create_field_space(c.ctx);
    allocator = c.runtime->create_field_allocator(c.ctx, fs);
    field_id_vec.clear();
  }

  FdSpace::~FdSpace()
  {
    ctx.runtime->destroy_field_space(ctx.ctx, fs);
  }

  /////////////////////////////////////////////////////////////
  // UserTask 
  /////////////////////////////////////////////////////////////
  
  UserTask::UserTask(const char* name="default") : task_name(name)
  {
    id = globalId++; // still not sure about this
  }
  
  UserTask::~UserTask(void)
  {
  }
  
  /////////////////////////////////////////////////////////////
  // TaskRuntime 
  /////////////////////////////////////////////////////////////
  
  TaskRuntime::TaskRuntime(void)
  {
    user_task_map.clear();
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
      printf("can not find task %p\n", (void*)func_ptr);
      return NULL;
    }
  }
  
}; // namespace LegionSimplified
