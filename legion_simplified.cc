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
    field_space = c.runtime->create_field_space(c.ctx);
    allocator = c.runtime->create_field_allocator(c.ctx, field_space);
    field_id_vector.clear();
  }

  FdSpace::~FdSpace(void)
  {
    ctx.runtime->destroy_field_space(ctx.ctx, field_space);
  }
  
  void FdSpace::add_field(size_t size, field_id_t field_id)
  {
    allocator.allocate_field(size,field_id);
    field_id_vector.push_back(field_id);
  }
  
  /////////////////////////////////////////////////////////////
  // BaseRegionImpl
  ///////////////////////////////////////////////////////////// 
  
  //----------------------------------public-------------------------------------
  BaseRegionImpl::BaseRegionImpl(const context ctx) :
    ctx(ctx), 
    logical_region(Legion::LogicalRegion::NO_REGION), 
    logical_partition(Legion::LogicalPartition::NO_PART), 
    logical_region_parent(Legion::LogicalRegion::NO_REGION),
    is_mapped(PR_NOT_MAPPED)
  {
    DEBUG_PRINT((2, "BaseRegionImpl(shared_ptr) constructor %p\n", this));
    logical_region = Legion::LogicalRegion::NO_REGION;
    logical_partition = Legion::LogicalPartition::NO_PART;
    logical_region_parent = Legion::LogicalRegion::NO_REGION;
    is_mapped = PR_NOT_MAPPED;
    field_id_vector.clear();
    accessor_map.clear();
    domain = Legion::Domain::NO_DOMAIN;
  }
  
  BaseRegionImpl::~BaseRegionImpl(void)
  {
    DEBUG_PRINT((2, "BaseRegionImpl(shared_ptr) destructor %p\n", this));
    logical_region = Legion::LogicalRegion::NO_REGION;
    logical_partition = Legion::LogicalPartition::NO_PART;
    logical_region_parent = Legion::LogicalRegion::NO_REGION;
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
    
    if (is_mapped == PR_INLINE_MAPPED) {
      unmap_physical_region();
    }
  }
  
  void BaseRegionImpl::init_accessor_map(void)
  {
    assert(field_id_vector.size() != 0);
    std::vector<field_id_t>::iterator it; 
    for (it = field_id_vector.begin(); it < field_id_vector.end(); it++) {
      DEBUG_PRINT((4, "BaseRegionImpl %p, init_accessor_map for fid %d\n", this, *it));
      unsigned char *null_ptr = NULL;
      accessor_map.insert(std::make_pair(*it, null_ptr));
    }
  }
  
  bool BaseRegionImpl::is_valid(void)
  {
    if (field_id_vector.size() == 0) {
      return false;
    } else {
      return true;
    }
  }
  
  void BaseRegionImpl::unmap_physical_region(void)
  {
    if (physical_region.is_mapped()) {
      ctx.runtime->unmap_region(ctx.ctx, physical_region);
      is_mapped = PR_NOT_MAPPED;
      
      DEBUG_PRINT((4, "BaseRegionImpl %p, unmap region inline\n", this));
    }
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
    
    context ctx;
    std::shared_ptr<BaseRegionImpl> empty_region = std::make_shared<BaseRegionImpl>(ctx);
    assert(empty_region->is_valid() == false);
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
  
  std::vector<std::shared_ptr<BaseRegionImpl>>::const_iterator TaskRuntime::check_inline_map_conflict(std::shared_ptr<BaseRegionImpl> &new_region)
  {
    return inline_map_region_vector.cbegin();
  }
  
  void TaskRuntime::add_inline_map(std::shared_ptr<BaseRegionImpl> &new_region)
  {
    inline_map_region_vector.push_back(new_region);
  }
  
  void TaskRuntime::remove_inline_map(std::vector<std::shared_ptr<BaseRegionImpl>>::const_iterator it)
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
