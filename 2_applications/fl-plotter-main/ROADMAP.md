# TODO THIS SHOULD ALL BE LOADED FROM THE YAML OF EACH FOLDER, architecture probably needs to be changed so that it takes in the path to the data, the output dir you want and the path to the YAML

# TODO yeah lets just set the max at the 95 percentile of the data as an outlier detection mechannis this should be clcualtedprogramtically like belw which is great


# THIS SHOULD COME FROM YAML


# OLD

notes from me
- honestly the plotboddy should have a style_dir on init, and have getters for teh style to then pass into seaborn
- the general design principle is that "all my context is handled by the buddy"

# FL-Plotter Development Roadmap

## Current Status
The project has a working implementation with `gs_plot_helpers.py` and `gs.mplstyle` that demonstrates proper chart styling for Goldman Sachs-style visualizations. The `ai_power_demand_growth.py` example shows good usage patterns.

## Goals
- Create a lightweight, self-contained plotting library
- Minimize external dependencies and system-wide installations
- Provide a clean class-based interface for chart creation
- Support flexible styling without requiring system matplotlib style installation

## Implementation Plan

### Phase 1: Core Refactoring
1. **Rename and restructure** `gs_plot_helpers.py` → `plot_buddy.py`
   - Convert standalone functions to `PlotBuddy` class methods
   - Keep current functionality intact during transition
   - Maintain backward compatibility initially

2. **Generalize logo functionality**
   - Rename `add_gs_logo()` → `add_logo()`
   - Make logo positioning and sizing more flexible
   - Support different logo formats and fallback behavior

3. **Enhanced title system**
   - Improve `add_chart_titles()` to better handle subtitles
   - Add support for multiple subtitle lines
   - Better positioning and formatting options

### Phase 2: Style System Improvements
4. **Local style loading**
   - Implement local `.mplstyle` file loading without system installation
   - Add `PlotBuddy.load_style_from_file()` method
   - Support style customization and overrides

5. **Style management**
   - Create style presets (GS style, minimal style, etc.)
   - Enable style switching within the same session
   - Maintain style consistency across multiple charts

### Phase 3: Class Design
6. **PlotBuddy class architecture**
   ```python
   class PlotBuddy:
       def __init__(self, style_dir_path = "whatever user needs to set it every time"):
           # Load style, set defaults,
       
       def setup_figure(self, figsize=None):
           # Create figure with proper styling
       
       def add_logo(self, fig, logo_path, position='bottom-right'):
           # Generic logo placement
       
       def add_titles(self, ax, title, subtitle=None, subtitle2=None):
           # Enhanced title system
       
       def create_legend(self, ax, **kwargs):
           # Consistent legend styling
   ```

### Phase 4: Example Updates
7. **Update existing examples**
   - Modify `ai_power_demand_growth.py` to use new `PlotBuddy` class
   - Demonstrate best practices and new features
   - Maintain current output quality

8. **Documentation and examples**
   - Create usage examples for common chart types
   - Document class methods and parameters
   - Add style customization examples

## Technical Decisions

### Lightweight Approach
- Use `mplstyle` for most styling (colors, fonts, grids)
- Keep helper functions minimal and focused
- Avoid heavy dependencies beyond matplotlib/pandas

### Local Style Loading
- Implement `plt.style.use()` with local file paths
- No system-wide matplotlib style installation required
- Enable style bundling with the package

### Backward Compatibility
- Maintain function-based API alongside class-based API initially
- Gradual migration path for existing code
- Clear deprecation warnings for old patterns

## Success Criteria
- ✅ No system-wide installations required
- ✅ Clean class-based interface
- ✅ Maintains current chart quality and styling
- ✅ Easy to use and extend
- ✅ Lightweight and fast

## Next Steps
1. Begin Phase 1 with `PlotBuddy` class creation
2. Implement local style loading
3. Test with existing `ai_power_demand_growth.py` example, CALL THIS A NEW FILE and go at it
4. Iterate and refine based on usage patterns