#include "legion_simplified.h"

namespace LegionSimplified {

  /////////////////////////////////////////////////////////////
  // IdxSpace
  /////////////////////////////////////////////////////////////
  
  template <size_t DIM>
  IdxSpace<DIM>::IdxSpace(const context& c, Point<DIM> p) : ctx(c)
  {
    rect = Rect<DIM>(Point<DIM>::ZEROES(), p-Point<DIM>::ONES());
    std::cout << "ispace set rect to be from " << Point<DIM>::ZEROES() << " to " << rect.hi << std::endl; 
    is = c.runtime->create_index_space(c.ctx, Legion::Domain::from_rect<DIM>(rect)); 
  }
  
  template <size_t DIM>
  IdxSpace<DIM>::~IdxSpace()
  {
    ctx.runtime->destroy_index_space(ctx.ctx, is);
  }
  
  /////////////////////////////////////////////////////////////
  // Fdpace
  /////////////////////////////////////////////////////////////
  
  template <typename T>
  void FdSpace::add_field(field_id_t fid)
  {
    allocator.allocate_field(sizeof(T),fid);
    field_id_vec.push_back(fid);
  }
}; // namespace LegionSimplified