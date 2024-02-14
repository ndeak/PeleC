# Example paraview batch script that creates an image of electron number density and
# reduced electric field, side by side. See NOTE locations for usage options

# trace generated using paraview version 5.8.1
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'AMReX/BoxLib Grid Reader'
# NOTE: Set your plotfile here
plt03000 = AMReXBoxLibGridReader(FileNames=['/scratch1/04361/ndeak/pelec_data-7lev-300K-1atm-parabolicPinsFP-1p25mmh-50um-2p5mm-sigmoidPulse-11kVanode-2nsdt-2nsrt-1nspt-1MHz-PACMorrowIonNoCap-zeroGradBC-adiabatic-impnEDiff/plt07316/'])
plt03000.EnableCaching = 0
plt03000.Level = 1
plt03000.PointArrayStatus = []
plt03000.CellArrayStatus = []

# Properties modified on plt03000
# NOTE: Control number of AMR levels loaded
plt03000.Level = 8
# NOTE: PeleC variables to be loaded in
plt03000.CellArrayStatus = ['Efieldx', 'Efieldy', 'Efieldz', 'n(E)', 'vfrac']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# NOTE: Set paraview domain resolution (horizontal x vertical, should match resolution for images generated at end)
renderView1.ViewSize = [2048, 1792]

# get layout
layout1 = GetLayout()

# show data in view
plt03000Display = Show(plt03000, renderView1, 'AMRRepresentation')

# trace defaults for the display properties.
plt03000Display.Representation = 'Outline'
plt03000Display.ColorArrayName = [None, '']
plt03000Display.LookupTable = None
plt03000Display.MapScalars = 1
plt03000Display.MultiComponentsMapping = 0
plt03000Display.InterpolateScalarsBeforeMapping = 1
plt03000Display.Opacity = 1.0
plt03000Display.PointSize = 2.0
plt03000Display.LineWidth = 1.0
plt03000Display.RenderLinesAsTubes = 0
plt03000Display.RenderPointsAsSpheres = 0
plt03000Display.Interpolation = 'Gouraud'
plt03000Display.Specular = 0.0
plt03000Display.SpecularColor = [1.0, 1.0, 1.0]
plt03000Display.SpecularPower = 100.0
plt03000Display.Luminosity = 0.0
plt03000Display.Ambient = 0.0
plt03000Display.Diffuse = 1.0
plt03000Display.Roughness = 0.3
plt03000Display.Metallic = 0.0
plt03000Display.Texture = None
plt03000Display.RepeatTextures = 1
plt03000Display.InterpolateTextures = 0
plt03000Display.SeamlessU = 0
plt03000Display.SeamlessV = 0
plt03000Display.UseMipmapTextures = 0
plt03000Display.BaseColorTexture = None
plt03000Display.NormalTexture = None
plt03000Display.NormalScale = 1.0
plt03000Display.MaterialTexture = None
plt03000Display.OcclusionStrength = 1.0
plt03000Display.EmissiveTexture = None
plt03000Display.EmissiveFactor = [1.0, 1.0, 1.0]
plt03000Display.FlipTextures = 0
plt03000Display.BackfaceRepresentation = 'Follow Frontface'
plt03000Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
plt03000Display.BackfaceOpacity = 1.0
plt03000Display.Position = [0.0, 0.0, 0.0]
plt03000Display.Scale = [1.0, 1.0, 1.0]
plt03000Display.Orientation = [0.0, 0.0, 0.0]
plt03000Display.Origin = [0.0, 0.0, 0.0]
plt03000Display.Pickable = 1
plt03000Display.Triangulate = 0
plt03000Display.UseShaderReplacements = 0
plt03000Display.ShaderReplacements = ''
plt03000Display.NonlinearSubdivisionLevel = 1
plt03000Display.UseDataPartitions = 0
plt03000Display.OSPRayUseScaleArray = 0
plt03000Display.OSPRayScaleArray = ''
plt03000Display.OSPRayScaleFunction = 'PiecewiseFunction'
plt03000Display.OSPRayMaterial = 'None'
plt03000Display.Orient = 0
plt03000Display.OrientationMode = 'Direction'
plt03000Display.SelectOrientationVectors = 'None'
plt03000Display.Scaling = 0
plt03000Display.ScaleMode = 'No Data Scaling Off'
plt03000Display.ScaleFactor = 0.17500000000000002
plt03000Display.SelectScaleArray = 'None'
plt03000Display.GlyphType = 'Arrow'
plt03000Display.UseGlyphTable = 0
plt03000Display.GlyphTableIndexArray = 'None'
plt03000Display.UseCompositeGlyphTable = 0
plt03000Display.UseGlyphCullingAndLOD = 0
plt03000Display.LODValues = []
plt03000Display.ColorByLODIndex = 0
plt03000Display.GaussianRadius = 0.00875
plt03000Display.ShaderPreset = 'Sphere'
plt03000Display.CustomTriangleScale = 3
plt03000Display.CustomShader = """ // This custom shader code define a gaussian blur
 // Please take a look into vtkSMPointGaussianRepresentation.cxx
 // for other custom shader examples
 //VTK::Color::Impl
   float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
   float gaussian = exp(-0.5*dist2);
   opacity = opacity*gaussian;
"""
plt03000Display.Emissive = 0
plt03000Display.ScaleByArray = 0
plt03000Display.SetScaleArray = [None, '']
plt03000Display.ScaleArrayComponent = 0
plt03000Display.UseScaleFunction = 1
plt03000Display.ScaleTransferFunction = 'PiecewiseFunction'
plt03000Display.OpacityByArray = 0
plt03000Display.OpacityArray = [None, '']
plt03000Display.OpacityArrayComponent = 0
plt03000Display.OpacityTransferFunction = 'PiecewiseFunction'
plt03000Display.DataAxesGrid = 'GridAxesRepresentation'
plt03000Display.SelectionCellLabelBold = 0
plt03000Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
plt03000Display.SelectionCellLabelFontFamily = 'Arial'
plt03000Display.SelectionCellLabelFontFile = ''
plt03000Display.SelectionCellLabelFontSize = 18
plt03000Display.SelectionCellLabelItalic = 0
plt03000Display.SelectionCellLabelJustification = 'Left'
plt03000Display.SelectionCellLabelOpacity = 1.0
plt03000Display.SelectionCellLabelShadow = 0
plt03000Display.SelectionPointLabelBold = 0
plt03000Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
plt03000Display.SelectionPointLabelFontFamily = 'Arial'
plt03000Display.SelectionPointLabelFontFile = ''
plt03000Display.SelectionPointLabelFontSize = 18
plt03000Display.SelectionPointLabelItalic = 0
plt03000Display.SelectionPointLabelJustification = 'Left'
plt03000Display.SelectionPointLabelOpacity = 1.0
plt03000Display.SelectionPointLabelShadow = 0
plt03000Display.PolarAxes = 'PolarAxesRepresentation'
plt03000Display.ScalarOpacityUnitDistance = 0.005265163702939256
plt03000Display.ScalarOpacityFunction = None
plt03000Display.VolumeRenderingMode = 'Smart'
plt03000Display.ResamplingMode = 'Over Data Bounds'
plt03000Display.StreamingRequestSize = 10
plt03000Display.NumberOfSamples = [64, 128, 64]
plt03000Display.Shade = 0

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
plt03000Display.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
plt03000Display.OSPRayScaleFunction.UseLogScale = 0

# init the 'Arrow' selected for 'GlyphType'
plt03000Display.GlyphType.TipResolution = 6
plt03000Display.GlyphType.TipRadius = 0.1
plt03000Display.GlyphType.TipLength = 0.35
plt03000Display.GlyphType.ShaftResolution = 6
plt03000Display.GlyphType.ShaftRadius = 0.03
plt03000Display.GlyphType.Invert = 0

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
plt03000Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
plt03000Display.ScaleTransferFunction.UseLogScale = 0

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
plt03000Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
plt03000Display.OpacityTransferFunction.UseLogScale = 0

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
plt03000Display.DataAxesGrid.XTitle = 'X Axis'
plt03000Display.DataAxesGrid.YTitle = 'Y Axis'
plt03000Display.DataAxesGrid.ZTitle = 'Z Axis'
plt03000Display.DataAxesGrid.XTitleFontFamily = 'Arial'
plt03000Display.DataAxesGrid.XTitleFontFile = ''
plt03000Display.DataAxesGrid.XTitleBold = 0
plt03000Display.DataAxesGrid.XTitleItalic = 0
plt03000Display.DataAxesGrid.XTitleFontSize = 12
plt03000Display.DataAxesGrid.XTitleShadow = 0
plt03000Display.DataAxesGrid.XTitleOpacity = 1.0
plt03000Display.DataAxesGrid.YTitleFontFamily = 'Arial'
plt03000Display.DataAxesGrid.YTitleFontFile = ''
plt03000Display.DataAxesGrid.YTitleBold = 0
plt03000Display.DataAxesGrid.YTitleItalic = 0
plt03000Display.DataAxesGrid.YTitleFontSize = 12
plt03000Display.DataAxesGrid.YTitleShadow = 0
plt03000Display.DataAxesGrid.YTitleOpacity = 1.0
plt03000Display.DataAxesGrid.ZTitleFontFamily = 'Arial'
plt03000Display.DataAxesGrid.ZTitleFontFile = ''
plt03000Display.DataAxesGrid.ZTitleBold = 0
plt03000Display.DataAxesGrid.ZTitleItalic = 0
plt03000Display.DataAxesGrid.ZTitleFontSize = 12
plt03000Display.DataAxesGrid.ZTitleShadow = 0
plt03000Display.DataAxesGrid.ZTitleOpacity = 1.0
plt03000Display.DataAxesGrid.FacesToRender = 63
plt03000Display.DataAxesGrid.CullBackface = 0
plt03000Display.DataAxesGrid.CullFrontface = 1
plt03000Display.DataAxesGrid.ShowGrid = 0
plt03000Display.DataAxesGrid.ShowEdges = 1
plt03000Display.DataAxesGrid.ShowTicks = 1
plt03000Display.DataAxesGrid.LabelUniqueEdgesOnly = 1
plt03000Display.DataAxesGrid.AxesToLabel = 63
plt03000Display.DataAxesGrid.XLabelFontFamily = 'Arial'
plt03000Display.DataAxesGrid.XLabelFontFile = ''
plt03000Display.DataAxesGrid.XLabelBold = 0
plt03000Display.DataAxesGrid.XLabelItalic = 0
plt03000Display.DataAxesGrid.XLabelFontSize = 12
plt03000Display.DataAxesGrid.XLabelShadow = 0
plt03000Display.DataAxesGrid.XLabelOpacity = 1.0
plt03000Display.DataAxesGrid.YLabelFontFamily = 'Arial'
plt03000Display.DataAxesGrid.YLabelFontFile = ''
plt03000Display.DataAxesGrid.YLabelBold = 0
plt03000Display.DataAxesGrid.YLabelItalic = 0
plt03000Display.DataAxesGrid.YLabelFontSize = 12
plt03000Display.DataAxesGrid.YLabelShadow = 0
plt03000Display.DataAxesGrid.YLabelOpacity = 1.0
plt03000Display.DataAxesGrid.ZLabelFontFamily = 'Arial'
plt03000Display.DataAxesGrid.ZLabelFontFile = ''
plt03000Display.DataAxesGrid.ZLabelBold = 0
plt03000Display.DataAxesGrid.ZLabelItalic = 0
plt03000Display.DataAxesGrid.ZLabelFontSize = 12
plt03000Display.DataAxesGrid.ZLabelShadow = 0
plt03000Display.DataAxesGrid.ZLabelOpacity = 1.0
plt03000Display.DataAxesGrid.XAxisNotation = 'Mixed'
plt03000Display.DataAxesGrid.XAxisPrecision = 2
plt03000Display.DataAxesGrid.XAxisUseCustomLabels = 0
plt03000Display.DataAxesGrid.XAxisLabels = []
plt03000Display.DataAxesGrid.YAxisNotation = 'Mixed'
plt03000Display.DataAxesGrid.YAxisPrecision = 2
plt03000Display.DataAxesGrid.YAxisUseCustomLabels = 0
plt03000Display.DataAxesGrid.YAxisLabels = []
plt03000Display.DataAxesGrid.ZAxisNotation = 'Mixed'
plt03000Display.DataAxesGrid.ZAxisPrecision = 2
plt03000Display.DataAxesGrid.ZAxisUseCustomLabels = 0
plt03000Display.DataAxesGrid.ZAxisLabels = []
plt03000Display.DataAxesGrid.UseCustomBounds = 0
plt03000Display.DataAxesGrid.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
plt03000Display.PolarAxes.Visibility = 0
plt03000Display.PolarAxes.Translation = [0.0, 0.0, 0.0]
plt03000Display.PolarAxes.Scale = [1.0, 1.0, 1.0]
plt03000Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]
plt03000Display.PolarAxes.EnableCustomBounds = [0, 0, 0]
plt03000Display.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
plt03000Display.PolarAxes.EnableCustomRange = 0
plt03000Display.PolarAxes.CustomRange = [0.0, 1.0]
plt03000Display.PolarAxes.PolarAxisVisibility = 1
plt03000Display.PolarAxes.RadialAxesVisibility = 1
plt03000Display.PolarAxes.DrawRadialGridlines = 1
plt03000Display.PolarAxes.PolarArcsVisibility = 1
plt03000Display.PolarAxes.DrawPolarArcsGridlines = 1
plt03000Display.PolarAxes.NumberOfRadialAxes = 0
plt03000Display.PolarAxes.AutoSubdividePolarAxis = 1
plt03000Display.PolarAxes.NumberOfPolarAxis = 0
plt03000Display.PolarAxes.MinimumRadius = 0.0
plt03000Display.PolarAxes.MinimumAngle = 0.0
plt03000Display.PolarAxes.MaximumAngle = 90.0
plt03000Display.PolarAxes.RadialAxesOriginToPolarAxis = 1
plt03000Display.PolarAxes.Ratio = 1.0
plt03000Display.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
plt03000Display.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
plt03000Display.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
plt03000Display.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
plt03000Display.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
plt03000Display.PolarAxes.PolarAxisTitleVisibility = 1
plt03000Display.PolarAxes.PolarAxisTitle = 'Radial Distance'
plt03000Display.PolarAxes.PolarAxisTitleLocation = 'Bottom'
plt03000Display.PolarAxes.PolarLabelVisibility = 1
plt03000Display.PolarAxes.PolarLabelFormat = '%-#6.3g'
plt03000Display.PolarAxes.PolarLabelExponentLocation = 'Labels'
plt03000Display.PolarAxes.RadialLabelVisibility = 1
plt03000Display.PolarAxes.RadialLabelFormat = '%-#3.1f'
plt03000Display.PolarAxes.RadialLabelLocation = 'Bottom'
plt03000Display.PolarAxes.RadialUnitsVisibility = 1
plt03000Display.PolarAxes.ScreenSize = 10.0
plt03000Display.PolarAxes.PolarAxisTitleOpacity = 1.0
plt03000Display.PolarAxes.PolarAxisTitleFontFamily = 'Arial'
plt03000Display.PolarAxes.PolarAxisTitleFontFile = ''
plt03000Display.PolarAxes.PolarAxisTitleBold = 0
plt03000Display.PolarAxes.PolarAxisTitleItalic = 0
plt03000Display.PolarAxes.PolarAxisTitleShadow = 0
plt03000Display.PolarAxes.PolarAxisTitleFontSize = 12
plt03000Display.PolarAxes.PolarAxisLabelOpacity = 1.0
plt03000Display.PolarAxes.PolarAxisLabelFontFamily = 'Arial'
plt03000Display.PolarAxes.PolarAxisLabelFontFile = ''
plt03000Display.PolarAxes.PolarAxisLabelBold = 0
plt03000Display.PolarAxes.PolarAxisLabelItalic = 0
plt03000Display.PolarAxes.PolarAxisLabelShadow = 0
plt03000Display.PolarAxes.PolarAxisLabelFontSize = 12
plt03000Display.PolarAxes.LastRadialAxisTextOpacity = 1.0
plt03000Display.PolarAxes.LastRadialAxisTextFontFamily = 'Arial'
plt03000Display.PolarAxes.LastRadialAxisTextFontFile = ''
plt03000Display.PolarAxes.LastRadialAxisTextBold = 0
plt03000Display.PolarAxes.LastRadialAxisTextItalic = 0
plt03000Display.PolarAxes.LastRadialAxisTextShadow = 0
plt03000Display.PolarAxes.LastRadialAxisTextFontSize = 12
plt03000Display.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
plt03000Display.PolarAxes.SecondaryRadialAxesTextFontFamily = 'Arial'
plt03000Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''
plt03000Display.PolarAxes.SecondaryRadialAxesTextBold = 0
plt03000Display.PolarAxes.SecondaryRadialAxesTextItalic = 0
plt03000Display.PolarAxes.SecondaryRadialAxesTextShadow = 0
plt03000Display.PolarAxes.SecondaryRadialAxesTextFontSize = 12
plt03000Display.PolarAxes.EnableDistanceLOD = 1
plt03000Display.PolarAxes.DistanceLODThreshold = 0.7
plt03000Display.PolarAxes.EnableViewAngleLOD = 1
plt03000Display.PolarAxes.ViewAngleLODThreshold = 0.7
plt03000Display.PolarAxes.SmallestVisiblePolarAngle = 0.5
plt03000Display.PolarAxes.PolarTicksVisibility = 1
plt03000Display.PolarAxes.ArcTicksOriginToPolarAxis = 1
plt03000Display.PolarAxes.TickLocation = 'Both'
plt03000Display.PolarAxes.AxisTickVisibility = 1
plt03000Display.PolarAxes.AxisMinorTickVisibility = 0
plt03000Display.PolarAxes.ArcTickVisibility = 1
plt03000Display.PolarAxes.ArcMinorTickVisibility = 0
plt03000Display.PolarAxes.DeltaAngleMajor = 10.0
plt03000Display.PolarAxes.DeltaAngleMinor = 5.0
plt03000Display.PolarAxes.PolarAxisMajorTickSize = 0.0
plt03000Display.PolarAxes.PolarAxisTickRatioSize = 0.3
plt03000Display.PolarAxes.PolarAxisMajorTickThickness = 1.0
plt03000Display.PolarAxes.PolarAxisTickRatioThickness = 0.5
plt03000Display.PolarAxes.LastRadialAxisMajorTickSize = 0.0
plt03000Display.PolarAxes.LastRadialAxisTickRatioSize = 0.3
plt03000Display.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
plt03000Display.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
plt03000Display.PolarAxes.ArcMajorTickSize = 0.0
plt03000Display.PolarAxes.ArcTickRatioSize = 0.3
plt03000Display.PolarAxes.ArcMajorTickThickness = 1.0
plt03000Display.PolarAxes.ArcTickRatioThickness = 0.5
plt03000Display.PolarAxes.Use2DMode = 0
plt03000Display.PolarAxes.UseLogAxis = 0

# reset view to fit data
renderView1.ResetCamera()

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Threshold'
threshold1 = Threshold(Input=plt03000)
threshold1.Scalars = ['CELLS', 'Efieldx']
threshold1.ThresholdRange = [-1508218475365.2095, 873093765032.3282]
threshold1.AllScalars = 1
threshold1.UseContinuousCellRange = 0
threshold1.Invert = 0

# Properties modified on threshold1
threshold1.Scalars = ['CELLS', 'vfrac']
threshold1.ThresholdRange = [1e-15, 1.0]

# show data in view
threshold1Display = Show(threshold1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
threshold1Display.Representation = 'Surface'
threshold1Display.ColorArrayName = [None, '']
threshold1Display.LookupTable = None
threshold1Display.MapScalars = 1
threshold1Display.MultiComponentsMapping = 0
threshold1Display.InterpolateScalarsBeforeMapping = 1
threshold1Display.Opacity = 1.0
threshold1Display.PointSize = 2.0
threshold1Display.LineWidth = 1.0
threshold1Display.RenderLinesAsTubes = 0
threshold1Display.RenderPointsAsSpheres = 0
threshold1Display.Interpolation = 'Gouraud'
threshold1Display.Specular = 0.0
threshold1Display.SpecularColor = [1.0, 1.0, 1.0]
threshold1Display.SpecularPower = 100.0
threshold1Display.Luminosity = 0.0
threshold1Display.Ambient = 0.0
threshold1Display.Diffuse = 1.0
threshold1Display.Roughness = 0.3
threshold1Display.Metallic = 0.0
threshold1Display.Texture = None
threshold1Display.RepeatTextures = 1
threshold1Display.InterpolateTextures = 0
threshold1Display.SeamlessU = 0
threshold1Display.SeamlessV = 0
threshold1Display.UseMipmapTextures = 0
threshold1Display.BaseColorTexture = None
threshold1Display.NormalTexture = None
threshold1Display.NormalScale = 1.0
threshold1Display.MaterialTexture = None
threshold1Display.OcclusionStrength = 1.0
threshold1Display.EmissiveTexture = None
threshold1Display.EmissiveFactor = [1.0, 1.0, 1.0]
threshold1Display.FlipTextures = 0
threshold1Display.BackfaceRepresentation = 'Follow Frontface'
threshold1Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
threshold1Display.BackfaceOpacity = 1.0
threshold1Display.Position = [0.0, 0.0, 0.0]
threshold1Display.Scale = [1.0, 1.0, 1.0]
threshold1Display.Orientation = [0.0, 0.0, 0.0]
threshold1Display.Origin = [0.0, 0.0, 0.0]
threshold1Display.Pickable = 1
threshold1Display.Triangulate = 0
threshold1Display.UseShaderReplacements = 0
threshold1Display.ShaderReplacements = ''
threshold1Display.NonlinearSubdivisionLevel = 1
threshold1Display.UseDataPartitions = 0
threshold1Display.OSPRayUseScaleArray = 0
threshold1Display.OSPRayScaleArray = ''
threshold1Display.OSPRayScaleFunction = 'PiecewiseFunction'
threshold1Display.OSPRayMaterial = 'None'
threshold1Display.Orient = 0
threshold1Display.OrientationMode = 'Direction'
threshold1Display.SelectOrientationVectors = 'None'
threshold1Display.Scaling = 0
threshold1Display.ScaleMode = 'No Data Scaling Off'
threshold1Display.ScaleFactor = 0.17500000000000002
threshold1Display.SelectScaleArray = 'None'
threshold1Display.GlyphType = 'Arrow'
threshold1Display.UseGlyphTable = 0
threshold1Display.GlyphTableIndexArray = 'None'
threshold1Display.UseCompositeGlyphTable = 0
threshold1Display.UseGlyphCullingAndLOD = 0
threshold1Display.LODValues = []
threshold1Display.ColorByLODIndex = 0
threshold1Display.GaussianRadius = 0.00875
threshold1Display.ShaderPreset = 'Sphere'
threshold1Display.CustomTriangleScale = 3
threshold1Display.CustomShader = """ // This custom shader code define a gaussian blur
 // Please take a look into vtkSMPointGaussianRepresentation.cxx
 // for other custom shader examples
 //VTK::Color::Impl
   float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
   float gaussian = exp(-0.5*dist2);
   opacity = opacity*gaussian;
"""
threshold1Display.Emissive = 0
threshold1Display.ScaleByArray = 0
threshold1Display.SetScaleArray = [None, '']
threshold1Display.ScaleArrayComponent = 0
threshold1Display.UseScaleFunction = 1
threshold1Display.ScaleTransferFunction = 'PiecewiseFunction'
threshold1Display.OpacityByArray = 0
threshold1Display.OpacityArray = [None, '']
threshold1Display.OpacityArrayComponent = 0
threshold1Display.OpacityTransferFunction = 'PiecewiseFunction'
threshold1Display.DataAxesGrid = 'GridAxesRepresentation'
threshold1Display.SelectionCellLabelBold = 0
threshold1Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
threshold1Display.SelectionCellLabelFontFamily = 'Arial'
threshold1Display.SelectionCellLabelFontFile = ''
threshold1Display.SelectionCellLabelFontSize = 18
threshold1Display.SelectionCellLabelItalic = 0
threshold1Display.SelectionCellLabelJustification = 'Left'
threshold1Display.SelectionCellLabelOpacity = 1.0
threshold1Display.SelectionCellLabelShadow = 0
threshold1Display.SelectionPointLabelBold = 0
threshold1Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
threshold1Display.SelectionPointLabelFontFamily = 'Arial'
threshold1Display.SelectionPointLabelFontFile = ''
threshold1Display.SelectionPointLabelFontSize = 18
threshold1Display.SelectionPointLabelItalic = 0
threshold1Display.SelectionPointLabelJustification = 'Left'
threshold1Display.SelectionPointLabelOpacity = 1.0
threshold1Display.SelectionPointLabelShadow = 0
threshold1Display.PolarAxes = 'PolarAxesRepresentation'
threshold1Display.ScalarOpacityFunction = None
threshold1Display.ScalarOpacityUnitDistance = 0.006657677148146404
threshold1Display.ExtractedBlockIndex = 1
threshold1Display.SelectMapper = 'Projected tetra'
threshold1Display.SamplingDimensions = [128, 128, 128]
threshold1Display.UseFloatingPointFrameBuffer = 1

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
threshold1Display.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
threshold1Display.OSPRayScaleFunction.UseLogScale = 0

# init the 'Arrow' selected for 'GlyphType'
threshold1Display.GlyphType.TipResolution = 6
threshold1Display.GlyphType.TipRadius = 0.1
threshold1Display.GlyphType.TipLength = 0.35
threshold1Display.GlyphType.ShaftResolution = 6
threshold1Display.GlyphType.ShaftRadius = 0.03
threshold1Display.GlyphType.Invert = 0

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
threshold1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
threshold1Display.ScaleTransferFunction.UseLogScale = 0

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
threshold1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
threshold1Display.OpacityTransferFunction.UseLogScale = 0

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
threshold1Display.DataAxesGrid.XTitle = 'X Axis'
threshold1Display.DataAxesGrid.YTitle = 'Y Axis'
threshold1Display.DataAxesGrid.ZTitle = 'Z Axis'
threshold1Display.DataAxesGrid.XTitleFontFamily = 'Arial'
threshold1Display.DataAxesGrid.XTitleFontFile = ''
threshold1Display.DataAxesGrid.XTitleBold = 0
threshold1Display.DataAxesGrid.XTitleItalic = 0
threshold1Display.DataAxesGrid.XTitleFontSize = 12
threshold1Display.DataAxesGrid.XTitleShadow = 0
threshold1Display.DataAxesGrid.XTitleOpacity = 1.0
threshold1Display.DataAxesGrid.YTitleFontFamily = 'Arial'
threshold1Display.DataAxesGrid.YTitleFontFile = ''
threshold1Display.DataAxesGrid.YTitleBold = 0
threshold1Display.DataAxesGrid.YTitleItalic = 0
threshold1Display.DataAxesGrid.YTitleFontSize = 12
threshold1Display.DataAxesGrid.YTitleShadow = 0
threshold1Display.DataAxesGrid.YTitleOpacity = 1.0
threshold1Display.DataAxesGrid.ZTitleFontFamily = 'Arial'
threshold1Display.DataAxesGrid.ZTitleFontFile = ''
threshold1Display.DataAxesGrid.ZTitleBold = 0
threshold1Display.DataAxesGrid.ZTitleItalic = 0
threshold1Display.DataAxesGrid.ZTitleFontSize = 12
threshold1Display.DataAxesGrid.ZTitleShadow = 0
threshold1Display.DataAxesGrid.ZTitleOpacity = 1.0
threshold1Display.DataAxesGrid.FacesToRender = 63
threshold1Display.DataAxesGrid.CullBackface = 0
threshold1Display.DataAxesGrid.CullFrontface = 1
threshold1Display.DataAxesGrid.ShowGrid = 0
threshold1Display.DataAxesGrid.ShowEdges = 1
threshold1Display.DataAxesGrid.ShowTicks = 1
threshold1Display.DataAxesGrid.LabelUniqueEdgesOnly = 1
threshold1Display.DataAxesGrid.AxesToLabel = 63
threshold1Display.DataAxesGrid.XLabelFontFamily = 'Arial'
threshold1Display.DataAxesGrid.XLabelFontFile = ''
threshold1Display.DataAxesGrid.XLabelBold = 0
threshold1Display.DataAxesGrid.XLabelItalic = 0
threshold1Display.DataAxesGrid.XLabelFontSize = 12
threshold1Display.DataAxesGrid.XLabelShadow = 0
threshold1Display.DataAxesGrid.XLabelOpacity = 1.0
threshold1Display.DataAxesGrid.YLabelFontFamily = 'Arial'
threshold1Display.DataAxesGrid.YLabelFontFile = ''
threshold1Display.DataAxesGrid.YLabelBold = 0
threshold1Display.DataAxesGrid.YLabelItalic = 0
threshold1Display.DataAxesGrid.YLabelFontSize = 12
threshold1Display.DataAxesGrid.YLabelShadow = 0
threshold1Display.DataAxesGrid.YLabelOpacity = 1.0
threshold1Display.DataAxesGrid.ZLabelFontFamily = 'Arial'
threshold1Display.DataAxesGrid.ZLabelFontFile = ''
threshold1Display.DataAxesGrid.ZLabelBold = 0
threshold1Display.DataAxesGrid.ZLabelItalic = 0
threshold1Display.DataAxesGrid.ZLabelFontSize = 12
threshold1Display.DataAxesGrid.ZLabelShadow = 0
threshold1Display.DataAxesGrid.ZLabelOpacity = 1.0
threshold1Display.DataAxesGrid.XAxisNotation = 'Mixed'
threshold1Display.DataAxesGrid.XAxisPrecision = 2
threshold1Display.DataAxesGrid.XAxisUseCustomLabels = 0
threshold1Display.DataAxesGrid.XAxisLabels = []
threshold1Display.DataAxesGrid.YAxisNotation = 'Mixed'
threshold1Display.DataAxesGrid.YAxisPrecision = 2
threshold1Display.DataAxesGrid.YAxisUseCustomLabels = 0
threshold1Display.DataAxesGrid.YAxisLabels = []
threshold1Display.DataAxesGrid.ZAxisNotation = 'Mixed'
threshold1Display.DataAxesGrid.ZAxisPrecision = 2
threshold1Display.DataAxesGrid.ZAxisUseCustomLabels = 0
threshold1Display.DataAxesGrid.ZAxisLabels = []
threshold1Display.DataAxesGrid.UseCustomBounds = 0
threshold1Display.DataAxesGrid.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
threshold1Display.PolarAxes.Visibility = 0
threshold1Display.PolarAxes.Translation = [0.0, 0.0, 0.0]
threshold1Display.PolarAxes.Scale = [1.0, 1.0, 1.0]
threshold1Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]
threshold1Display.PolarAxes.EnableCustomBounds = [0, 0, 0]
threshold1Display.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
threshold1Display.PolarAxes.EnableCustomRange = 0
threshold1Display.PolarAxes.CustomRange = [0.0, 1.0]
threshold1Display.PolarAxes.PolarAxisVisibility = 1
threshold1Display.PolarAxes.RadialAxesVisibility = 1
threshold1Display.PolarAxes.DrawRadialGridlines = 1
threshold1Display.PolarAxes.PolarArcsVisibility = 1
threshold1Display.PolarAxes.DrawPolarArcsGridlines = 1
threshold1Display.PolarAxes.NumberOfRadialAxes = 0
threshold1Display.PolarAxes.AutoSubdividePolarAxis = 1
threshold1Display.PolarAxes.NumberOfPolarAxis = 0
threshold1Display.PolarAxes.MinimumRadius = 0.0
threshold1Display.PolarAxes.MinimumAngle = 0.0
threshold1Display.PolarAxes.MaximumAngle = 90.0
threshold1Display.PolarAxes.RadialAxesOriginToPolarAxis = 1
threshold1Display.PolarAxes.Ratio = 1.0
threshold1Display.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
threshold1Display.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
threshold1Display.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
threshold1Display.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
threshold1Display.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
threshold1Display.PolarAxes.PolarAxisTitleVisibility = 1
threshold1Display.PolarAxes.PolarAxisTitle = 'Radial Distance'
threshold1Display.PolarAxes.PolarAxisTitleLocation = 'Bottom'
threshold1Display.PolarAxes.PolarLabelVisibility = 1
threshold1Display.PolarAxes.PolarLabelFormat = '%-#6.3g'
threshold1Display.PolarAxes.PolarLabelExponentLocation = 'Labels'
threshold1Display.PolarAxes.RadialLabelVisibility = 1
threshold1Display.PolarAxes.RadialLabelFormat = '%-#3.1f'
threshold1Display.PolarAxes.RadialLabelLocation = 'Bottom'
threshold1Display.PolarAxes.RadialUnitsVisibility = 1
threshold1Display.PolarAxes.ScreenSize = 10.0
threshold1Display.PolarAxes.PolarAxisTitleOpacity = 1.0
threshold1Display.PolarAxes.PolarAxisTitleFontFamily = 'Arial'
threshold1Display.PolarAxes.PolarAxisTitleFontFile = ''
threshold1Display.PolarAxes.PolarAxisTitleBold = 0
threshold1Display.PolarAxes.PolarAxisTitleItalic = 0
threshold1Display.PolarAxes.PolarAxisTitleShadow = 0
threshold1Display.PolarAxes.PolarAxisTitleFontSize = 12
threshold1Display.PolarAxes.PolarAxisLabelOpacity = 1.0
threshold1Display.PolarAxes.PolarAxisLabelFontFamily = 'Arial'
threshold1Display.PolarAxes.PolarAxisLabelFontFile = ''
threshold1Display.PolarAxes.PolarAxisLabelBold = 0
threshold1Display.PolarAxes.PolarAxisLabelItalic = 0
threshold1Display.PolarAxes.PolarAxisLabelShadow = 0
threshold1Display.PolarAxes.PolarAxisLabelFontSize = 12
threshold1Display.PolarAxes.LastRadialAxisTextOpacity = 1.0
threshold1Display.PolarAxes.LastRadialAxisTextFontFamily = 'Arial'
threshold1Display.PolarAxes.LastRadialAxisTextFontFile = ''
threshold1Display.PolarAxes.LastRadialAxisTextBold = 0
threshold1Display.PolarAxes.LastRadialAxisTextItalic = 0
threshold1Display.PolarAxes.LastRadialAxisTextShadow = 0
threshold1Display.PolarAxes.LastRadialAxisTextFontSize = 12
threshold1Display.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
threshold1Display.PolarAxes.SecondaryRadialAxesTextFontFamily = 'Arial'
threshold1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''
threshold1Display.PolarAxes.SecondaryRadialAxesTextBold = 0
threshold1Display.PolarAxes.SecondaryRadialAxesTextItalic = 0
threshold1Display.PolarAxes.SecondaryRadialAxesTextShadow = 0
threshold1Display.PolarAxes.SecondaryRadialAxesTextFontSize = 12
threshold1Display.PolarAxes.EnableDistanceLOD = 1
threshold1Display.PolarAxes.DistanceLODThreshold = 0.7
threshold1Display.PolarAxes.EnableViewAngleLOD = 1
threshold1Display.PolarAxes.ViewAngleLODThreshold = 0.7
threshold1Display.PolarAxes.SmallestVisiblePolarAngle = 0.5
threshold1Display.PolarAxes.PolarTicksVisibility = 1
threshold1Display.PolarAxes.ArcTicksOriginToPolarAxis = 1
threshold1Display.PolarAxes.TickLocation = 'Both'
threshold1Display.PolarAxes.AxisTickVisibility = 1
threshold1Display.PolarAxes.AxisMinorTickVisibility = 0
threshold1Display.PolarAxes.ArcTickVisibility = 1
threshold1Display.PolarAxes.ArcMinorTickVisibility = 0
threshold1Display.PolarAxes.DeltaAngleMajor = 10.0
threshold1Display.PolarAxes.DeltaAngleMinor = 5.0
threshold1Display.PolarAxes.PolarAxisMajorTickSize = 0.0
threshold1Display.PolarAxes.PolarAxisTickRatioSize = 0.3
threshold1Display.PolarAxes.PolarAxisMajorTickThickness = 1.0
threshold1Display.PolarAxes.PolarAxisTickRatioThickness = 0.5
threshold1Display.PolarAxes.LastRadialAxisMajorTickSize = 0.0
threshold1Display.PolarAxes.LastRadialAxisTickRatioSize = 0.3
threshold1Display.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
threshold1Display.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
threshold1Display.PolarAxes.ArcMajorTickSize = 0.0
threshold1Display.PolarAxes.ArcTickRatioSize = 0.3
threshold1Display.PolarAxes.ArcMajorTickThickness = 1.0
threshold1Display.PolarAxes.ArcTickRatioThickness = 0.5
threshold1Display.PolarAxes.Use2DMode = 0
threshold1Display.PolarAxes.UseLogAxis = 0

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(threshold1Display, ('FIELD', 'vtkBlockColors'))

# show color bar/color legend
threshold1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'vtkBlockColors'
vtkBlockColorsLUT = GetColorTransferFunction('vtkBlockColors')

# get opacity transfer function/opacity map for 'vtkBlockColors'
vtkBlockColorsPWF = GetOpacityTransferFunction('vtkBlockColors')

# hide data in view
Hide(plt03000, renderView1)

# create a new 'Slice'
slice1 = Slice(Input=threshold1)
slice1.SliceType = 'Plane'
slice1.HyperTreeGridSlicer = 'Plane'
slice1.UseDual = 0
slice1.Crinkleslice = 0
slice1.Triangulatetheslice = 1
slice1.Mergeduplicatedpointsintheslice = 1
slice1.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Origin = [0.5, 0.875, 0.5]
slice1.SliceType.Normal = [1.0, 0.0, 0.0]
slice1.SliceType.Offset = 0.0

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice1.HyperTreeGridSlicer.Origin = [0.5, 0.875, 0.5]
slice1.HyperTreeGridSlicer.Normal = [1.0, 0.0, 0.0]
slice1.HyperTreeGridSlicer.Offset = 0.0

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice1.SliceType)

# Properties modified on slice1.SliceType
# NOTE: Change E/N the slice original and normal direction here
slice1.SliceType.Origin = [0.5, 0.875, 1.0]
slice1.SliceType.Normal = [0.0, 0.0, 1.0]

# show data in view
slice1Display = Show(slice1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
slice1Display.Representation = 'Surface'
slice1Display.ColorArrayName = [None, '']
slice1Display.LookupTable = None
slice1Display.MapScalars = 1
slice1Display.MultiComponentsMapping = 0
slice1Display.InterpolateScalarsBeforeMapping = 1
slice1Display.Opacity = 1.0
slice1Display.PointSize = 2.0
slice1Display.LineWidth = 1.0
slice1Display.RenderLinesAsTubes = 0
slice1Display.RenderPointsAsSpheres = 0
slice1Display.Interpolation = 'Gouraud'
slice1Display.Specular = 0.0
slice1Display.SpecularColor = [1.0, 1.0, 1.0]
slice1Display.SpecularPower = 100.0
slice1Display.Luminosity = 0.0
slice1Display.Ambient = 0.0
slice1Display.Diffuse = 1.0
slice1Display.Roughness = 0.3
slice1Display.Metallic = 0.0
slice1Display.Texture = None
slice1Display.RepeatTextures = 1
slice1Display.InterpolateTextures = 0
slice1Display.SeamlessU = 0
slice1Display.SeamlessV = 0
slice1Display.UseMipmapTextures = 0
slice1Display.BaseColorTexture = None
slice1Display.NormalTexture = None
slice1Display.NormalScale = 1.0
slice1Display.MaterialTexture = None
slice1Display.OcclusionStrength = 1.0
slice1Display.EmissiveTexture = None
slice1Display.EmissiveFactor = [1.0, 1.0, 1.0]
slice1Display.FlipTextures = 0
slice1Display.BackfaceRepresentation = 'Follow Frontface'
slice1Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
slice1Display.BackfaceOpacity = 1.0
slice1Display.Position = [0.0, 0.0, 0.0]
slice1Display.Scale = [1.0, 1.0, 1.0]
slice1Display.Orientation = [0.0, 0.0, 0.0]
slice1Display.Origin = [0.0, 0.0, 0.0]
slice1Display.Pickable = 1
slice1Display.Triangulate = 0
slice1Display.UseShaderReplacements = 0
slice1Display.ShaderReplacements = ''
slice1Display.NonlinearSubdivisionLevel = 1
slice1Display.UseDataPartitions = 0
slice1Display.OSPRayUseScaleArray = 0
slice1Display.OSPRayScaleArray = ''
slice1Display.OSPRayScaleFunction = 'PiecewiseFunction'
slice1Display.OSPRayMaterial = 'None'
slice1Display.Orient = 0
slice1Display.OrientationMode = 'Direction'
slice1Display.SelectOrientationVectors = 'None'
slice1Display.Scaling = 0
slice1Display.ScaleMode = 'No Data Scaling Off'
slice1Display.ScaleFactor = 0.17500000000000002
slice1Display.SelectScaleArray = 'None'
slice1Display.GlyphType = 'Arrow'
slice1Display.UseGlyphTable = 0
slice1Display.GlyphTableIndexArray = 'None'
slice1Display.UseCompositeGlyphTable = 0
slice1Display.UseGlyphCullingAndLOD = 0
slice1Display.LODValues = []
slice1Display.ColorByLODIndex = 0
slice1Display.GaussianRadius = 0.00875
slice1Display.ShaderPreset = 'Sphere'
slice1Display.CustomTriangleScale = 3
slice1Display.CustomShader = """ // This custom shader code define a gaussian blur
 // Please take a look into vtkSMPointGaussianRepresentation.cxx
 // for other custom shader examples
 //VTK::Color::Impl
   float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
   float gaussian = exp(-0.5*dist2);
   opacity = opacity*gaussian;
"""
slice1Display.Emissive = 0
slice1Display.ScaleByArray = 0
slice1Display.SetScaleArray = [None, '']
slice1Display.ScaleArrayComponent = 0
slice1Display.UseScaleFunction = 1
slice1Display.ScaleTransferFunction = 'PiecewiseFunction'
slice1Display.OpacityByArray = 0
slice1Display.OpacityArray = [None, '']
slice1Display.OpacityArrayComponent = 0
slice1Display.OpacityTransferFunction = 'PiecewiseFunction'
slice1Display.DataAxesGrid = 'GridAxesRepresentation'
slice1Display.SelectionCellLabelBold = 0
slice1Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
slice1Display.SelectionCellLabelFontFamily = 'Arial'
slice1Display.SelectionCellLabelFontFile = ''
slice1Display.SelectionCellLabelFontSize = 18
slice1Display.SelectionCellLabelItalic = 0
slice1Display.SelectionCellLabelJustification = 'Left'
slice1Display.SelectionCellLabelOpacity = 1.0
slice1Display.SelectionCellLabelShadow = 0
slice1Display.SelectionPointLabelBold = 0
slice1Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
slice1Display.SelectionPointLabelFontFamily = 'Arial'
slice1Display.SelectionPointLabelFontFile = ''
slice1Display.SelectionPointLabelFontSize = 18
slice1Display.SelectionPointLabelItalic = 0
slice1Display.SelectionPointLabelJustification = 'Left'
slice1Display.SelectionPointLabelOpacity = 1.0
slice1Display.SelectionPointLabelShadow = 0
slice1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
slice1Display.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
slice1Display.OSPRayScaleFunction.UseLogScale = 0

# init the 'Arrow' selected for 'GlyphType'
slice1Display.GlyphType.TipResolution = 6
slice1Display.GlyphType.TipRadius = 0.1
slice1Display.GlyphType.TipLength = 0.35
slice1Display.GlyphType.ShaftResolution = 6
slice1Display.GlyphType.ShaftRadius = 0.03
slice1Display.GlyphType.Invert = 0

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
slice1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
slice1Display.ScaleTransferFunction.UseLogScale = 0

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
slice1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
slice1Display.OpacityTransferFunction.UseLogScale = 0

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
slice1Display.DataAxesGrid.XTitle = 'X Axis'
slice1Display.DataAxesGrid.YTitle = 'Y Axis'
slice1Display.DataAxesGrid.ZTitle = 'Z Axis'
slice1Display.DataAxesGrid.XTitleFontFamily = 'Arial'
slice1Display.DataAxesGrid.XTitleFontFile = ''
slice1Display.DataAxesGrid.XTitleBold = 0
slice1Display.DataAxesGrid.XTitleItalic = 0
slice1Display.DataAxesGrid.XTitleFontSize = 12
slice1Display.DataAxesGrid.XTitleShadow = 0
slice1Display.DataAxesGrid.XTitleOpacity = 1.0
slice1Display.DataAxesGrid.YTitleFontFamily = 'Arial'
slice1Display.DataAxesGrid.YTitleFontFile = ''
slice1Display.DataAxesGrid.YTitleBold = 0
slice1Display.DataAxesGrid.YTitleItalic = 0
slice1Display.DataAxesGrid.YTitleFontSize = 12
slice1Display.DataAxesGrid.YTitleShadow = 0
slice1Display.DataAxesGrid.YTitleOpacity = 1.0
slice1Display.DataAxesGrid.ZTitleFontFamily = 'Arial'
slice1Display.DataAxesGrid.ZTitleFontFile = ''
slice1Display.DataAxesGrid.ZTitleBold = 0
slice1Display.DataAxesGrid.ZTitleItalic = 0
slice1Display.DataAxesGrid.ZTitleFontSize = 12
slice1Display.DataAxesGrid.ZTitleShadow = 0
slice1Display.DataAxesGrid.ZTitleOpacity = 1.0
slice1Display.DataAxesGrid.FacesToRender = 63
slice1Display.DataAxesGrid.CullBackface = 0
slice1Display.DataAxesGrid.CullFrontface = 1
slice1Display.DataAxesGrid.ShowGrid = 0
slice1Display.DataAxesGrid.ShowEdges = 1
slice1Display.DataAxesGrid.ShowTicks = 1
slice1Display.DataAxesGrid.LabelUniqueEdgesOnly = 1
slice1Display.DataAxesGrid.AxesToLabel = 63
slice1Display.DataAxesGrid.XLabelFontFamily = 'Arial'
slice1Display.DataAxesGrid.XLabelFontFile = ''
slice1Display.DataAxesGrid.XLabelBold = 0
slice1Display.DataAxesGrid.XLabelItalic = 0
slice1Display.DataAxesGrid.XLabelFontSize = 12
slice1Display.DataAxesGrid.XLabelShadow = 0
slice1Display.DataAxesGrid.XLabelOpacity = 1.0
slice1Display.DataAxesGrid.YLabelFontFamily = 'Arial'
slice1Display.DataAxesGrid.YLabelFontFile = ''
slice1Display.DataAxesGrid.YLabelBold = 0
slice1Display.DataAxesGrid.YLabelItalic = 0
slice1Display.DataAxesGrid.YLabelFontSize = 12
slice1Display.DataAxesGrid.YLabelShadow = 0
slice1Display.DataAxesGrid.YLabelOpacity = 1.0
slice1Display.DataAxesGrid.ZLabelFontFamily = 'Arial'
slice1Display.DataAxesGrid.ZLabelFontFile = ''
slice1Display.DataAxesGrid.ZLabelBold = 0
slice1Display.DataAxesGrid.ZLabelItalic = 0
slice1Display.DataAxesGrid.ZLabelFontSize = 12
slice1Display.DataAxesGrid.ZLabelShadow = 0
slice1Display.DataAxesGrid.ZLabelOpacity = 1.0
slice1Display.DataAxesGrid.XAxisNotation = 'Mixed'
slice1Display.DataAxesGrid.XAxisPrecision = 2
slice1Display.DataAxesGrid.XAxisUseCustomLabels = 0
slice1Display.DataAxesGrid.XAxisLabels = []
slice1Display.DataAxesGrid.YAxisNotation = 'Mixed'
slice1Display.DataAxesGrid.YAxisPrecision = 2
slice1Display.DataAxesGrid.YAxisUseCustomLabels = 0
slice1Display.DataAxesGrid.YAxisLabels = []
slice1Display.DataAxesGrid.ZAxisNotation = 'Mixed'
slice1Display.DataAxesGrid.ZAxisPrecision = 2
slice1Display.DataAxesGrid.ZAxisUseCustomLabels = 0
slice1Display.DataAxesGrid.ZAxisLabels = []
slice1Display.DataAxesGrid.UseCustomBounds = 0
slice1Display.DataAxesGrid.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
slice1Display.PolarAxes.Visibility = 0
slice1Display.PolarAxes.Translation = [0.0, 0.0, 0.0]
slice1Display.PolarAxes.Scale = [1.0, 1.0, 1.0]
slice1Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]
slice1Display.PolarAxes.EnableCustomBounds = [0, 0, 0]
slice1Display.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
slice1Display.PolarAxes.EnableCustomRange = 0
slice1Display.PolarAxes.CustomRange = [0.0, 1.0]
slice1Display.PolarAxes.PolarAxisVisibility = 1
slice1Display.PolarAxes.RadialAxesVisibility = 1
slice1Display.PolarAxes.DrawRadialGridlines = 1
slice1Display.PolarAxes.PolarArcsVisibility = 1
slice1Display.PolarAxes.DrawPolarArcsGridlines = 1
slice1Display.PolarAxes.NumberOfRadialAxes = 0
slice1Display.PolarAxes.AutoSubdividePolarAxis = 1
slice1Display.PolarAxes.NumberOfPolarAxis = 0
slice1Display.PolarAxes.MinimumRadius = 0.0
slice1Display.PolarAxes.MinimumAngle = 0.0
slice1Display.PolarAxes.MaximumAngle = 90.0
slice1Display.PolarAxes.RadialAxesOriginToPolarAxis = 1
slice1Display.PolarAxes.Ratio = 1.0
slice1Display.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
slice1Display.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
slice1Display.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
slice1Display.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
slice1Display.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
slice1Display.PolarAxes.PolarAxisTitleVisibility = 1
slice1Display.PolarAxes.PolarAxisTitle = 'Radial Distance'
slice1Display.PolarAxes.PolarAxisTitleLocation = 'Bottom'
slice1Display.PolarAxes.PolarLabelVisibility = 1
slice1Display.PolarAxes.PolarLabelFormat = '%-#6.3g'
slice1Display.PolarAxes.PolarLabelExponentLocation = 'Labels'
slice1Display.PolarAxes.RadialLabelVisibility = 1
slice1Display.PolarAxes.RadialLabelFormat = '%-#3.1f'
slice1Display.PolarAxes.RadialLabelLocation = 'Bottom'
slice1Display.PolarAxes.RadialUnitsVisibility = 1
slice1Display.PolarAxes.ScreenSize = 10.0
slice1Display.PolarAxes.PolarAxisTitleOpacity = 1.0
slice1Display.PolarAxes.PolarAxisTitleFontFamily = 'Arial'
slice1Display.PolarAxes.PolarAxisTitleFontFile = ''
slice1Display.PolarAxes.PolarAxisTitleBold = 0
slice1Display.PolarAxes.PolarAxisTitleItalic = 0
slice1Display.PolarAxes.PolarAxisTitleShadow = 0
slice1Display.PolarAxes.PolarAxisTitleFontSize = 12
slice1Display.PolarAxes.PolarAxisLabelOpacity = 1.0
slice1Display.PolarAxes.PolarAxisLabelFontFamily = 'Arial'
slice1Display.PolarAxes.PolarAxisLabelFontFile = ''
slice1Display.PolarAxes.PolarAxisLabelBold = 0
slice1Display.PolarAxes.PolarAxisLabelItalic = 0
slice1Display.PolarAxes.PolarAxisLabelShadow = 0
slice1Display.PolarAxes.PolarAxisLabelFontSize = 12
slice1Display.PolarAxes.LastRadialAxisTextOpacity = 1.0
slice1Display.PolarAxes.LastRadialAxisTextFontFamily = 'Arial'
slice1Display.PolarAxes.LastRadialAxisTextFontFile = ''
slice1Display.PolarAxes.LastRadialAxisTextBold = 0
slice1Display.PolarAxes.LastRadialAxisTextItalic = 0
slice1Display.PolarAxes.LastRadialAxisTextShadow = 0
slice1Display.PolarAxes.LastRadialAxisTextFontSize = 12
slice1Display.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
slice1Display.PolarAxes.SecondaryRadialAxesTextFontFamily = 'Arial'
slice1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''
slice1Display.PolarAxes.SecondaryRadialAxesTextBold = 0
slice1Display.PolarAxes.SecondaryRadialAxesTextItalic = 0
slice1Display.PolarAxes.SecondaryRadialAxesTextShadow = 0
slice1Display.PolarAxes.SecondaryRadialAxesTextFontSize = 12
slice1Display.PolarAxes.EnableDistanceLOD = 1
slice1Display.PolarAxes.DistanceLODThreshold = 0.7
slice1Display.PolarAxes.EnableViewAngleLOD = 1
slice1Display.PolarAxes.ViewAngleLODThreshold = 0.7
slice1Display.PolarAxes.SmallestVisiblePolarAngle = 0.5
slice1Display.PolarAxes.PolarTicksVisibility = 1
slice1Display.PolarAxes.ArcTicksOriginToPolarAxis = 1
slice1Display.PolarAxes.TickLocation = 'Both'
slice1Display.PolarAxes.AxisTickVisibility = 1
slice1Display.PolarAxes.AxisMinorTickVisibility = 0
slice1Display.PolarAxes.ArcTickVisibility = 1
slice1Display.PolarAxes.ArcMinorTickVisibility = 0
slice1Display.PolarAxes.DeltaAngleMajor = 10.0
slice1Display.PolarAxes.DeltaAngleMinor = 5.0
slice1Display.PolarAxes.PolarAxisMajorTickSize = 0.0
slice1Display.PolarAxes.PolarAxisTickRatioSize = 0.3
slice1Display.PolarAxes.PolarAxisMajorTickThickness = 1.0
slice1Display.PolarAxes.PolarAxisTickRatioThickness = 0.5
slice1Display.PolarAxes.LastRadialAxisMajorTickSize = 0.0
slice1Display.PolarAxes.LastRadialAxisTickRatioSize = 0.3
slice1Display.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
slice1Display.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
slice1Display.PolarAxes.ArcMajorTickSize = 0.0
slice1Display.PolarAxes.ArcTickRatioSize = 0.3
slice1Display.PolarAxes.ArcMajorTickThickness = 1.0
slice1Display.PolarAxes.ArcTickRatioThickness = 0.5
slice1Display.PolarAxes.Use2DMode = 0
slice1Display.PolarAxes.UseLogAxis = 0

# hide data in view
Hide(threshold1, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(slice1Display, ('FIELD', 'vtkBlockColors'))

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# create a new 'Calculator'
calculator1 = Calculator(Input=slice1)
calculator1.AttributeType = 'Cell Data'
calculator1.CoordinateResults = 0
calculator1.ResultNormals = 0
calculator1.ResultTCoords = 0
calculator1.ResultArrayName = 'Result'
calculator1.Function = ''
calculator1.ReplaceInvalidResults = 1
calculator1.ReplacementValue = 0.0
calculator1.ResultArrayType = 'Double'

# Properties modified on calculator1
calculator1.ResultArrayName = 'E/N (Td)'
calculator1.Function = '(Efieldx*Efieldx + Efieldy*Efieldy + Efieldz*Efieldz)^.5 / (2.45e9)'

# show data in view
calculator1Display = Show(calculator1, renderView1, 'GeometryRepresentation')

# get color transfer function/color map for 'ENTd'
eNTdLUT = GetColorTransferFunction('ENTd')

# trace defaults for the display properties.
calculator1Display.Representation = 'Surface'
calculator1Display.ColorArrayName = ['CELLS', 'E/N (Td)']
calculator1Display.LookupTable = eNTdLUT
calculator1Display.MapScalars = 1
calculator1Display.MultiComponentsMapping = 0
calculator1Display.InterpolateScalarsBeforeMapping = 1
calculator1Display.Opacity = 1.0
calculator1Display.PointSize = 2.0
calculator1Display.LineWidth = 1.0
calculator1Display.RenderLinesAsTubes = 0
calculator1Display.RenderPointsAsSpheres = 0
calculator1Display.Interpolation = 'Gouraud'
calculator1Display.Specular = 0.0
calculator1Display.SpecularColor = [1.0, 1.0, 1.0]
calculator1Display.SpecularPower = 100.0
calculator1Display.Luminosity = 0.0
calculator1Display.Ambient = 0.0
calculator1Display.Diffuse = 1.0
calculator1Display.Roughness = 0.3
calculator1Display.Metallic = 0.0
calculator1Display.Texture = None
calculator1Display.RepeatTextures = 1
calculator1Display.InterpolateTextures = 0
calculator1Display.SeamlessU = 0
calculator1Display.SeamlessV = 0
calculator1Display.UseMipmapTextures = 0
calculator1Display.BaseColorTexture = None
calculator1Display.NormalTexture = None
calculator1Display.NormalScale = 1.0
calculator1Display.MaterialTexture = None
calculator1Display.OcclusionStrength = 1.0
calculator1Display.EmissiveTexture = None
calculator1Display.EmissiveFactor = [1.0, 1.0, 1.0]
calculator1Display.FlipTextures = 0
calculator1Display.BackfaceRepresentation = 'Follow Frontface'
calculator1Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
calculator1Display.BackfaceOpacity = 1.0
calculator1Display.Position = [0.0, 0.0, 0.0]
calculator1Display.Scale = [1.0, 1.0, 1.0]
calculator1Display.Orientation = [0.0, 0.0, 0.0]
calculator1Display.Origin = [0.0, 0.0, 0.0]
calculator1Display.Pickable = 1
calculator1Display.Triangulate = 0
calculator1Display.UseShaderReplacements = 0
calculator1Display.ShaderReplacements = ''
calculator1Display.NonlinearSubdivisionLevel = 1
calculator1Display.UseDataPartitions = 0
calculator1Display.OSPRayUseScaleArray = 0
calculator1Display.OSPRayScaleArray = ''
calculator1Display.OSPRayScaleFunction = 'PiecewiseFunction'
calculator1Display.OSPRayMaterial = 'None'
calculator1Display.Orient = 0
calculator1Display.OrientationMode = 'Direction'
calculator1Display.SelectOrientationVectors = 'None'
calculator1Display.Scaling = 0
calculator1Display.ScaleMode = 'No Data Scaling Off'
calculator1Display.ScaleFactor = 0.17500000000000002
calculator1Display.SelectScaleArray = 'E/N (Td)'
calculator1Display.GlyphType = 'Arrow'
calculator1Display.UseGlyphTable = 0
calculator1Display.GlyphTableIndexArray = 'E/N (Td)'
calculator1Display.UseCompositeGlyphTable = 0
calculator1Display.UseGlyphCullingAndLOD = 0
calculator1Display.LODValues = []
calculator1Display.ColorByLODIndex = 0
calculator1Display.GaussianRadius = 0.00875
calculator1Display.ShaderPreset = 'Sphere'
calculator1Display.CustomTriangleScale = 3
calculator1Display.CustomShader = """ // This custom shader code define a gaussian blur
 // Please take a look into vtkSMPointGaussianRepresentation.cxx
 // for other custom shader examples
 //VTK::Color::Impl
   float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
   float gaussian = exp(-0.5*dist2);
   opacity = opacity*gaussian;
"""
calculator1Display.Emissive = 0
calculator1Display.ScaleByArray = 0
calculator1Display.SetScaleArray = [None, '']
calculator1Display.ScaleArrayComponent = 0
calculator1Display.UseScaleFunction = 1
calculator1Display.ScaleTransferFunction = 'PiecewiseFunction'
calculator1Display.OpacityByArray = 0
calculator1Display.OpacityArray = [None, '']
calculator1Display.OpacityArrayComponent = 0
calculator1Display.OpacityTransferFunction = 'PiecewiseFunction'
calculator1Display.DataAxesGrid = 'GridAxesRepresentation'
calculator1Display.SelectionCellLabelBold = 0
calculator1Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
calculator1Display.SelectionCellLabelFontFamily = 'Arial'
calculator1Display.SelectionCellLabelFontFile = ''
calculator1Display.SelectionCellLabelFontSize = 18
calculator1Display.SelectionCellLabelItalic = 0
calculator1Display.SelectionCellLabelJustification = 'Left'
calculator1Display.SelectionCellLabelOpacity = 1.0
calculator1Display.SelectionCellLabelShadow = 0
calculator1Display.SelectionPointLabelBold = 0
calculator1Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
calculator1Display.SelectionPointLabelFontFamily = 'Arial'
calculator1Display.SelectionPointLabelFontFile = ''
calculator1Display.SelectionPointLabelFontSize = 18
calculator1Display.SelectionPointLabelItalic = 0
calculator1Display.SelectionPointLabelJustification = 'Left'
calculator1Display.SelectionPointLabelOpacity = 1.0
calculator1Display.SelectionPointLabelShadow = 0
calculator1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
calculator1Display.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
calculator1Display.OSPRayScaleFunction.UseLogScale = 0

# init the 'Arrow' selected for 'GlyphType'
calculator1Display.GlyphType.TipResolution = 6
calculator1Display.GlyphType.TipRadius = 0.1
calculator1Display.GlyphType.TipLength = 0.35
calculator1Display.GlyphType.ShaftResolution = 6
calculator1Display.GlyphType.ShaftRadius = 0.03
calculator1Display.GlyphType.Invert = 0

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
calculator1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
calculator1Display.ScaleTransferFunction.UseLogScale = 0

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
calculator1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
calculator1Display.OpacityTransferFunction.UseLogScale = 0

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
calculator1Display.DataAxesGrid.XTitle = 'X Axis'
calculator1Display.DataAxesGrid.YTitle = 'Y Axis'
calculator1Display.DataAxesGrid.ZTitle = 'Z Axis'
calculator1Display.DataAxesGrid.XTitleFontFamily = 'Arial'
calculator1Display.DataAxesGrid.XTitleFontFile = ''
calculator1Display.DataAxesGrid.XTitleBold = 0
calculator1Display.DataAxesGrid.XTitleItalic = 0
calculator1Display.DataAxesGrid.XTitleFontSize = 12
calculator1Display.DataAxesGrid.XTitleShadow = 0
calculator1Display.DataAxesGrid.XTitleOpacity = 1.0
calculator1Display.DataAxesGrid.YTitleFontFamily = 'Arial'
calculator1Display.DataAxesGrid.YTitleFontFile = ''
calculator1Display.DataAxesGrid.YTitleBold = 0
calculator1Display.DataAxesGrid.YTitleItalic = 0
calculator1Display.DataAxesGrid.YTitleFontSize = 12
calculator1Display.DataAxesGrid.YTitleShadow = 0
calculator1Display.DataAxesGrid.YTitleOpacity = 1.0
calculator1Display.DataAxesGrid.ZTitleFontFamily = 'Arial'
calculator1Display.DataAxesGrid.ZTitleFontFile = ''
calculator1Display.DataAxesGrid.ZTitleBold = 0
calculator1Display.DataAxesGrid.ZTitleItalic = 0
calculator1Display.DataAxesGrid.ZTitleFontSize = 12
calculator1Display.DataAxesGrid.ZTitleShadow = 0
calculator1Display.DataAxesGrid.ZTitleOpacity = 1.0
calculator1Display.DataAxesGrid.FacesToRender = 63
calculator1Display.DataAxesGrid.CullBackface = 0
calculator1Display.DataAxesGrid.CullFrontface = 1
calculator1Display.DataAxesGrid.ShowGrid = 0
calculator1Display.DataAxesGrid.ShowEdges = 1
calculator1Display.DataAxesGrid.ShowTicks = 1
calculator1Display.DataAxesGrid.LabelUniqueEdgesOnly = 1
calculator1Display.DataAxesGrid.AxesToLabel = 63
calculator1Display.DataAxesGrid.XLabelFontFamily = 'Arial'
calculator1Display.DataAxesGrid.XLabelFontFile = ''
calculator1Display.DataAxesGrid.XLabelBold = 0
calculator1Display.DataAxesGrid.XLabelItalic = 0
calculator1Display.DataAxesGrid.XLabelFontSize = 12
calculator1Display.DataAxesGrid.XLabelShadow = 0
calculator1Display.DataAxesGrid.XLabelOpacity = 1.0
calculator1Display.DataAxesGrid.YLabelFontFamily = 'Arial'
calculator1Display.DataAxesGrid.YLabelFontFile = ''
calculator1Display.DataAxesGrid.YLabelBold = 0
calculator1Display.DataAxesGrid.YLabelItalic = 0
calculator1Display.DataAxesGrid.YLabelFontSize = 12
calculator1Display.DataAxesGrid.YLabelShadow = 0
calculator1Display.DataAxesGrid.YLabelOpacity = 1.0
calculator1Display.DataAxesGrid.ZLabelFontFamily = 'Arial'
calculator1Display.DataAxesGrid.ZLabelFontFile = ''
calculator1Display.DataAxesGrid.ZLabelBold = 0
calculator1Display.DataAxesGrid.ZLabelItalic = 0
calculator1Display.DataAxesGrid.ZLabelFontSize = 12
calculator1Display.DataAxesGrid.ZLabelShadow = 0
calculator1Display.DataAxesGrid.ZLabelOpacity = 1.0
calculator1Display.DataAxesGrid.XAxisNotation = 'Mixed'
calculator1Display.DataAxesGrid.XAxisPrecision = 2
calculator1Display.DataAxesGrid.XAxisUseCustomLabels = 0
calculator1Display.DataAxesGrid.XAxisLabels = []
calculator1Display.DataAxesGrid.YAxisNotation = 'Mixed'
calculator1Display.DataAxesGrid.YAxisPrecision = 2
calculator1Display.DataAxesGrid.YAxisUseCustomLabels = 0
calculator1Display.DataAxesGrid.YAxisLabels = []
calculator1Display.DataAxesGrid.ZAxisNotation = 'Mixed'
calculator1Display.DataAxesGrid.ZAxisPrecision = 2
calculator1Display.DataAxesGrid.ZAxisUseCustomLabels = 0
calculator1Display.DataAxesGrid.ZAxisLabels = []
calculator1Display.DataAxesGrid.UseCustomBounds = 0
calculator1Display.DataAxesGrid.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
calculator1Display.PolarAxes.Visibility = 0
calculator1Display.PolarAxes.Translation = [0.0, 0.0, 0.0]
calculator1Display.PolarAxes.Scale = [1.0, 1.0, 1.0]
calculator1Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]
calculator1Display.PolarAxes.EnableCustomBounds = [0, 0, 0]
calculator1Display.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
calculator1Display.PolarAxes.EnableCustomRange = 0
calculator1Display.PolarAxes.CustomRange = [0.0, 1.0]
calculator1Display.PolarAxes.PolarAxisVisibility = 1
calculator1Display.PolarAxes.RadialAxesVisibility = 1
calculator1Display.PolarAxes.DrawRadialGridlines = 1
calculator1Display.PolarAxes.PolarArcsVisibility = 1
calculator1Display.PolarAxes.DrawPolarArcsGridlines = 1
calculator1Display.PolarAxes.NumberOfRadialAxes = 0
calculator1Display.PolarAxes.AutoSubdividePolarAxis = 1
calculator1Display.PolarAxes.NumberOfPolarAxis = 0
calculator1Display.PolarAxes.MinimumRadius = 0.0
calculator1Display.PolarAxes.MinimumAngle = 0.0
calculator1Display.PolarAxes.MaximumAngle = 90.0
calculator1Display.PolarAxes.RadialAxesOriginToPolarAxis = 1
calculator1Display.PolarAxes.Ratio = 1.0
calculator1Display.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
calculator1Display.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
calculator1Display.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
calculator1Display.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
calculator1Display.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
calculator1Display.PolarAxes.PolarAxisTitleVisibility = 1
calculator1Display.PolarAxes.PolarAxisTitle = 'Radial Distance'
calculator1Display.PolarAxes.PolarAxisTitleLocation = 'Bottom'
calculator1Display.PolarAxes.PolarLabelVisibility = 1
calculator1Display.PolarAxes.PolarLabelFormat = '%-#6.3g'
calculator1Display.PolarAxes.PolarLabelExponentLocation = 'Labels'
calculator1Display.PolarAxes.RadialLabelVisibility = 1
calculator1Display.PolarAxes.RadialLabelFormat = '%-#3.1f'
calculator1Display.PolarAxes.RadialLabelLocation = 'Bottom'
calculator1Display.PolarAxes.RadialUnitsVisibility = 1
calculator1Display.PolarAxes.ScreenSize = 10.0
calculator1Display.PolarAxes.PolarAxisTitleOpacity = 1.0
calculator1Display.PolarAxes.PolarAxisTitleFontFamily = 'Arial'
calculator1Display.PolarAxes.PolarAxisTitleFontFile = ''
calculator1Display.PolarAxes.PolarAxisTitleBold = 0
calculator1Display.PolarAxes.PolarAxisTitleItalic = 0
calculator1Display.PolarAxes.PolarAxisTitleShadow = 0
calculator1Display.PolarAxes.PolarAxisTitleFontSize = 12
calculator1Display.PolarAxes.PolarAxisLabelOpacity = 1.0
calculator1Display.PolarAxes.PolarAxisLabelFontFamily = 'Arial'
calculator1Display.PolarAxes.PolarAxisLabelFontFile = ''
calculator1Display.PolarAxes.PolarAxisLabelBold = 0
calculator1Display.PolarAxes.PolarAxisLabelItalic = 0
calculator1Display.PolarAxes.PolarAxisLabelShadow = 0
calculator1Display.PolarAxes.PolarAxisLabelFontSize = 12
calculator1Display.PolarAxes.LastRadialAxisTextOpacity = 1.0
calculator1Display.PolarAxes.LastRadialAxisTextFontFamily = 'Arial'
calculator1Display.PolarAxes.LastRadialAxisTextFontFile = ''
calculator1Display.PolarAxes.LastRadialAxisTextBold = 0
calculator1Display.PolarAxes.LastRadialAxisTextItalic = 0
calculator1Display.PolarAxes.LastRadialAxisTextShadow = 0
calculator1Display.PolarAxes.LastRadialAxisTextFontSize = 12
calculator1Display.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
calculator1Display.PolarAxes.SecondaryRadialAxesTextFontFamily = 'Arial'
calculator1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''
calculator1Display.PolarAxes.SecondaryRadialAxesTextBold = 0
calculator1Display.PolarAxes.SecondaryRadialAxesTextItalic = 0
calculator1Display.PolarAxes.SecondaryRadialAxesTextShadow = 0
calculator1Display.PolarAxes.SecondaryRadialAxesTextFontSize = 12
calculator1Display.PolarAxes.EnableDistanceLOD = 1
calculator1Display.PolarAxes.DistanceLODThreshold = 0.7
calculator1Display.PolarAxes.EnableViewAngleLOD = 1
calculator1Display.PolarAxes.ViewAngleLODThreshold = 0.7
calculator1Display.PolarAxes.SmallestVisiblePolarAngle = 0.5
calculator1Display.PolarAxes.PolarTicksVisibility = 1
calculator1Display.PolarAxes.ArcTicksOriginToPolarAxis = 1
calculator1Display.PolarAxes.TickLocation = 'Both'
calculator1Display.PolarAxes.AxisTickVisibility = 1
calculator1Display.PolarAxes.AxisMinorTickVisibility = 0
calculator1Display.PolarAxes.ArcTickVisibility = 1
calculator1Display.PolarAxes.ArcMinorTickVisibility = 0
calculator1Display.PolarAxes.DeltaAngleMajor = 10.0
calculator1Display.PolarAxes.DeltaAngleMinor = 5.0
calculator1Display.PolarAxes.PolarAxisMajorTickSize = 0.0
calculator1Display.PolarAxes.PolarAxisTickRatioSize = 0.3
calculator1Display.PolarAxes.PolarAxisMajorTickThickness = 1.0
calculator1Display.PolarAxes.PolarAxisTickRatioThickness = 0.5
calculator1Display.PolarAxes.LastRadialAxisMajorTickSize = 0.0
calculator1Display.PolarAxes.LastRadialAxisTickRatioSize = 0.3
calculator1Display.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
calculator1Display.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
calculator1Display.PolarAxes.ArcMajorTickSize = 0.0
calculator1Display.PolarAxes.ArcTickRatioSize = 0.3
calculator1Display.PolarAxes.ArcMajorTickThickness = 1.0
calculator1Display.PolarAxes.ArcTickRatioThickness = 0.5
calculator1Display.PolarAxes.Use2DMode = 0
calculator1Display.PolarAxes.UseLogAxis = 0

# hide data in view
Hide(slice1, renderView1)

# show color bar/color legend
calculator1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get opacity transfer function/opacity map for 'ENTd'
eNTdPWF = GetOpacityTransferFunction('ENTd')

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
# NOTE: Change E/N color palette here
# See https://www.paraview.org/Wiki/images/7/73/Luts.png for examples of color palettes
eNTdLUT.ApplyPreset('X Ray', True)

# get color legend/bar for eNTdLUT in view renderView1
eNTdLUTColorBar = GetScalarBar(eNTdLUT, renderView1)

# Properties modified on eNTdLUTColorBar
eNTdLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
eNTdLUTColorBar.LabelColor = [0.0, 0.0, 0.0]

# create a new 'Reflect'
reflect1 = Reflect(Input=calculator1)
reflect1.Plane = 'X Min'
reflect1.Center = 0.0
reflect1.CopyInput = 0
reflect1.FlipAllInputArrays = 1

# Properties modified on reflect1
reflect1.Plane = 'X Max'

# show data in view
reflect1Display = Show(reflect1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
reflect1Display.Representation = 'Surface'
reflect1Display.ColorArrayName = ['CELLS', 'E/N (Td)']
reflect1Display.LookupTable = eNTdLUT
reflect1Display.MapScalars = 1
reflect1Display.MultiComponentsMapping = 0
reflect1Display.InterpolateScalarsBeforeMapping = 1
reflect1Display.Opacity = 1.0
reflect1Display.PointSize = 2.0
reflect1Display.LineWidth = 1.0
reflect1Display.RenderLinesAsTubes = 0
reflect1Display.RenderPointsAsSpheres = 0
reflect1Display.Interpolation = 'Gouraud'
reflect1Display.Specular = 0.0
reflect1Display.SpecularColor = [1.0, 1.0, 1.0]
reflect1Display.SpecularPower = 100.0
reflect1Display.Luminosity = 0.0
reflect1Display.Ambient = 0.0
reflect1Display.Diffuse = 1.0
reflect1Display.Roughness = 0.3
reflect1Display.Metallic = 0.0
reflect1Display.Texture = None
reflect1Display.RepeatTextures = 1
reflect1Display.InterpolateTextures = 0
reflect1Display.SeamlessU = 0
reflect1Display.SeamlessV = 0
reflect1Display.UseMipmapTextures = 0
reflect1Display.BaseColorTexture = None
reflect1Display.NormalTexture = None
reflect1Display.NormalScale = 1.0
reflect1Display.MaterialTexture = None
reflect1Display.OcclusionStrength = 1.0
reflect1Display.EmissiveTexture = None
reflect1Display.EmissiveFactor = [1.0, 1.0, 1.0]
reflect1Display.FlipTextures = 0
reflect1Display.BackfaceRepresentation = 'Follow Frontface'
reflect1Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
reflect1Display.BackfaceOpacity = 1.0
reflect1Display.Position = [0.0, 0.0, 0.0]
reflect1Display.Scale = [1.0, 1.0, 1.0]
reflect1Display.Orientation = [0.0, 0.0, 0.0]
reflect1Display.Origin = [0.0, 0.0, 0.0]
reflect1Display.Pickable = 1
reflect1Display.Triangulate = 0
reflect1Display.UseShaderReplacements = 0
reflect1Display.ShaderReplacements = ''
reflect1Display.NonlinearSubdivisionLevel = 1
reflect1Display.UseDataPartitions = 0
reflect1Display.OSPRayUseScaleArray = 0
reflect1Display.OSPRayScaleArray = ''
reflect1Display.OSPRayScaleFunction = 'PiecewiseFunction'
reflect1Display.OSPRayMaterial = 'None'
reflect1Display.Orient = 0
reflect1Display.OrientationMode = 'Direction'
reflect1Display.SelectOrientationVectors = 'None'
reflect1Display.Scaling = 0
reflect1Display.ScaleMode = 'No Data Scaling Off'
reflect1Display.ScaleFactor = 0.2
reflect1Display.SelectScaleArray = 'E/N (Td)'
reflect1Display.GlyphType = 'Arrow'
reflect1Display.UseGlyphTable = 0
reflect1Display.GlyphTableIndexArray = 'E/N (Td)'
reflect1Display.UseCompositeGlyphTable = 0
reflect1Display.UseGlyphCullingAndLOD = 0
reflect1Display.LODValues = []
reflect1Display.ColorByLODIndex = 0
reflect1Display.GaussianRadius = 0.01
reflect1Display.ShaderPreset = 'Sphere'
reflect1Display.CustomTriangleScale = 3
reflect1Display.CustomShader = """ // This custom shader code define a gaussian blur
 // Please take a look into vtkSMPointGaussianRepresentation.cxx
 // for other custom shader examples
 //VTK::Color::Impl
   float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
   float gaussian = exp(-0.5*dist2);
   opacity = opacity*gaussian;
"""
reflect1Display.Emissive = 0
reflect1Display.ScaleByArray = 0
reflect1Display.SetScaleArray = [None, '']
reflect1Display.ScaleArrayComponent = 0
reflect1Display.UseScaleFunction = 1
reflect1Display.ScaleTransferFunction = 'PiecewiseFunction'
reflect1Display.OpacityByArray = 0
reflect1Display.OpacityArray = [None, '']
reflect1Display.OpacityArrayComponent = 0
reflect1Display.OpacityTransferFunction = 'PiecewiseFunction'
reflect1Display.DataAxesGrid = 'GridAxesRepresentation'
reflect1Display.SelectionCellLabelBold = 0
reflect1Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
reflect1Display.SelectionCellLabelFontFamily = 'Arial'
reflect1Display.SelectionCellLabelFontFile = ''
reflect1Display.SelectionCellLabelFontSize = 18
reflect1Display.SelectionCellLabelItalic = 0
reflect1Display.SelectionCellLabelJustification = 'Left'
reflect1Display.SelectionCellLabelOpacity = 1.0
reflect1Display.SelectionCellLabelShadow = 0
reflect1Display.SelectionPointLabelBold = 0
reflect1Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
reflect1Display.SelectionPointLabelFontFamily = 'Arial'
reflect1Display.SelectionPointLabelFontFile = ''
reflect1Display.SelectionPointLabelFontSize = 18
reflect1Display.SelectionPointLabelItalic = 0
reflect1Display.SelectionPointLabelJustification = 'Left'
reflect1Display.SelectionPointLabelOpacity = 1.0
reflect1Display.SelectionPointLabelShadow = 0
reflect1Display.PolarAxes = 'PolarAxesRepresentation'
reflect1Display.ScalarOpacityFunction = eNTdPWF
reflect1Display.ScalarOpacityUnitDistance = 0.03326779211925393
reflect1Display.ExtractedBlockIndex = 1
reflect1Display.SelectMapper = 'Projected tetra'
reflect1Display.SamplingDimensions = [128, 128, 128]
reflect1Display.UseFloatingPointFrameBuffer = 1

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
reflect1Display.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
reflect1Display.OSPRayScaleFunction.UseLogScale = 0

# init the 'Arrow' selected for 'GlyphType'
reflect1Display.GlyphType.TipResolution = 6
reflect1Display.GlyphType.TipRadius = 0.1
reflect1Display.GlyphType.TipLength = 0.35
reflect1Display.GlyphType.ShaftResolution = 6
reflect1Display.GlyphType.ShaftRadius = 0.03
reflect1Display.GlyphType.Invert = 0

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
reflect1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
reflect1Display.ScaleTransferFunction.UseLogScale = 0

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
reflect1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
reflect1Display.OpacityTransferFunction.UseLogScale = 0

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
reflect1Display.DataAxesGrid.XTitle = 'X Axis'
reflect1Display.DataAxesGrid.YTitle = 'Y Axis'
reflect1Display.DataAxesGrid.ZTitle = 'Z Axis'
reflect1Display.DataAxesGrid.XTitleFontFamily = 'Arial'
reflect1Display.DataAxesGrid.XTitleFontFile = ''
reflect1Display.DataAxesGrid.XTitleBold = 0
reflect1Display.DataAxesGrid.XTitleItalic = 0
reflect1Display.DataAxesGrid.XTitleFontSize = 12
reflect1Display.DataAxesGrid.XTitleShadow = 0
reflect1Display.DataAxesGrid.XTitleOpacity = 1.0
reflect1Display.DataAxesGrid.YTitleFontFamily = 'Arial'
reflect1Display.DataAxesGrid.YTitleFontFile = ''
reflect1Display.DataAxesGrid.YTitleBold = 0
reflect1Display.DataAxesGrid.YTitleItalic = 0
reflect1Display.DataAxesGrid.YTitleFontSize = 12
reflect1Display.DataAxesGrid.YTitleShadow = 0
reflect1Display.DataAxesGrid.YTitleOpacity = 1.0
reflect1Display.DataAxesGrid.ZTitleFontFamily = 'Arial'
reflect1Display.DataAxesGrid.ZTitleFontFile = ''
reflect1Display.DataAxesGrid.ZTitleBold = 0
reflect1Display.DataAxesGrid.ZTitleItalic = 0
reflect1Display.DataAxesGrid.ZTitleFontSize = 12
reflect1Display.DataAxesGrid.ZTitleShadow = 0
reflect1Display.DataAxesGrid.ZTitleOpacity = 1.0
reflect1Display.DataAxesGrid.FacesToRender = 63
reflect1Display.DataAxesGrid.CullBackface = 0
reflect1Display.DataAxesGrid.CullFrontface = 1
reflect1Display.DataAxesGrid.ShowGrid = 0
reflect1Display.DataAxesGrid.ShowEdges = 1
reflect1Display.DataAxesGrid.ShowTicks = 1
reflect1Display.DataAxesGrid.LabelUniqueEdgesOnly = 1
reflect1Display.DataAxesGrid.AxesToLabel = 63
reflect1Display.DataAxesGrid.XLabelFontFamily = 'Arial'
reflect1Display.DataAxesGrid.XLabelFontFile = ''
reflect1Display.DataAxesGrid.XLabelBold = 0
reflect1Display.DataAxesGrid.XLabelItalic = 0
reflect1Display.DataAxesGrid.XLabelFontSize = 12
reflect1Display.DataAxesGrid.XLabelShadow = 0
reflect1Display.DataAxesGrid.XLabelOpacity = 1.0
reflect1Display.DataAxesGrid.YLabelFontFamily = 'Arial'
reflect1Display.DataAxesGrid.YLabelFontFile = ''
reflect1Display.DataAxesGrid.YLabelBold = 0
reflect1Display.DataAxesGrid.YLabelItalic = 0
reflect1Display.DataAxesGrid.YLabelFontSize = 12
reflect1Display.DataAxesGrid.YLabelShadow = 0
reflect1Display.DataAxesGrid.YLabelOpacity = 1.0
reflect1Display.DataAxesGrid.ZLabelFontFamily = 'Arial'
reflect1Display.DataAxesGrid.ZLabelFontFile = ''
reflect1Display.DataAxesGrid.ZLabelBold = 0
reflect1Display.DataAxesGrid.ZLabelItalic = 0
reflect1Display.DataAxesGrid.ZLabelFontSize = 12
reflect1Display.DataAxesGrid.ZLabelShadow = 0
reflect1Display.DataAxesGrid.ZLabelOpacity = 1.0
reflect1Display.DataAxesGrid.XAxisNotation = 'Mixed'
reflect1Display.DataAxesGrid.XAxisPrecision = 2
reflect1Display.DataAxesGrid.XAxisUseCustomLabels = 0
reflect1Display.DataAxesGrid.XAxisLabels = []
reflect1Display.DataAxesGrid.YAxisNotation = 'Mixed'
reflect1Display.DataAxesGrid.YAxisPrecision = 2
reflect1Display.DataAxesGrid.YAxisUseCustomLabels = 0
reflect1Display.DataAxesGrid.YAxisLabels = []
reflect1Display.DataAxesGrid.ZAxisNotation = 'Mixed'
reflect1Display.DataAxesGrid.ZAxisPrecision = 2
reflect1Display.DataAxesGrid.ZAxisUseCustomLabels = 0
reflect1Display.DataAxesGrid.ZAxisLabels = []
reflect1Display.DataAxesGrid.UseCustomBounds = 0
reflect1Display.DataAxesGrid.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
reflect1Display.PolarAxes.Visibility = 0
reflect1Display.PolarAxes.Translation = [0.0, 0.0, 0.0]
reflect1Display.PolarAxes.Scale = [1.0, 1.0, 1.0]
reflect1Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]
reflect1Display.PolarAxes.EnableCustomBounds = [0, 0, 0]
reflect1Display.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
reflect1Display.PolarAxes.EnableCustomRange = 0
reflect1Display.PolarAxes.CustomRange = [0.0, 1.0]
reflect1Display.PolarAxes.PolarAxisVisibility = 1
reflect1Display.PolarAxes.RadialAxesVisibility = 1
reflect1Display.PolarAxes.DrawRadialGridlines = 1
reflect1Display.PolarAxes.PolarArcsVisibility = 1
reflect1Display.PolarAxes.DrawPolarArcsGridlines = 1
reflect1Display.PolarAxes.NumberOfRadialAxes = 0
reflect1Display.PolarAxes.AutoSubdividePolarAxis = 1
reflect1Display.PolarAxes.NumberOfPolarAxis = 0
reflect1Display.PolarAxes.MinimumRadius = 0.0
reflect1Display.PolarAxes.MinimumAngle = 0.0
reflect1Display.PolarAxes.MaximumAngle = 90.0
reflect1Display.PolarAxes.RadialAxesOriginToPolarAxis = 1
reflect1Display.PolarAxes.Ratio = 1.0
reflect1Display.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
reflect1Display.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
reflect1Display.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
reflect1Display.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
reflect1Display.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
reflect1Display.PolarAxes.PolarAxisTitleVisibility = 1
reflect1Display.PolarAxes.PolarAxisTitle = 'Radial Distance'
reflect1Display.PolarAxes.PolarAxisTitleLocation = 'Bottom'
reflect1Display.PolarAxes.PolarLabelVisibility = 1
reflect1Display.PolarAxes.PolarLabelFormat = '%-#6.3g'
reflect1Display.PolarAxes.PolarLabelExponentLocation = 'Labels'
reflect1Display.PolarAxes.RadialLabelVisibility = 1
reflect1Display.PolarAxes.RadialLabelFormat = '%-#3.1f'
reflect1Display.PolarAxes.RadialLabelLocation = 'Bottom'
reflect1Display.PolarAxes.RadialUnitsVisibility = 1
reflect1Display.PolarAxes.ScreenSize = 10.0
reflect1Display.PolarAxes.PolarAxisTitleOpacity = 1.0
reflect1Display.PolarAxes.PolarAxisTitleFontFamily = 'Arial'
reflect1Display.PolarAxes.PolarAxisTitleFontFile = ''
reflect1Display.PolarAxes.PolarAxisTitleBold = 0
reflect1Display.PolarAxes.PolarAxisTitleItalic = 0
reflect1Display.PolarAxes.PolarAxisTitleShadow = 0
reflect1Display.PolarAxes.PolarAxisTitleFontSize = 12
reflect1Display.PolarAxes.PolarAxisLabelOpacity = 1.0
reflect1Display.PolarAxes.PolarAxisLabelFontFamily = 'Arial'
reflect1Display.PolarAxes.PolarAxisLabelFontFile = ''
reflect1Display.PolarAxes.PolarAxisLabelBold = 0
reflect1Display.PolarAxes.PolarAxisLabelItalic = 0
reflect1Display.PolarAxes.PolarAxisLabelShadow = 0
reflect1Display.PolarAxes.PolarAxisLabelFontSize = 12
reflect1Display.PolarAxes.LastRadialAxisTextOpacity = 1.0
reflect1Display.PolarAxes.LastRadialAxisTextFontFamily = 'Arial'
reflect1Display.PolarAxes.LastRadialAxisTextFontFile = ''
reflect1Display.PolarAxes.LastRadialAxisTextBold = 0
reflect1Display.PolarAxes.LastRadialAxisTextItalic = 0
reflect1Display.PolarAxes.LastRadialAxisTextShadow = 0
reflect1Display.PolarAxes.LastRadialAxisTextFontSize = 12
reflect1Display.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
reflect1Display.PolarAxes.SecondaryRadialAxesTextFontFamily = 'Arial'
reflect1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''
reflect1Display.PolarAxes.SecondaryRadialAxesTextBold = 0
reflect1Display.PolarAxes.SecondaryRadialAxesTextItalic = 0
reflect1Display.PolarAxes.SecondaryRadialAxesTextShadow = 0
reflect1Display.PolarAxes.SecondaryRadialAxesTextFontSize = 12
reflect1Display.PolarAxes.EnableDistanceLOD = 1
reflect1Display.PolarAxes.DistanceLODThreshold = 0.7
reflect1Display.PolarAxes.EnableViewAngleLOD = 1
reflect1Display.PolarAxes.ViewAngleLODThreshold = 0.7
reflect1Display.PolarAxes.SmallestVisiblePolarAngle = 0.5
reflect1Display.PolarAxes.PolarTicksVisibility = 1
reflect1Display.PolarAxes.ArcTicksOriginToPolarAxis = 1
reflect1Display.PolarAxes.TickLocation = 'Both'
reflect1Display.PolarAxes.AxisTickVisibility = 1
reflect1Display.PolarAxes.AxisMinorTickVisibility = 0
reflect1Display.PolarAxes.ArcTickVisibility = 1
reflect1Display.PolarAxes.ArcMinorTickVisibility = 0
reflect1Display.PolarAxes.DeltaAngleMajor = 10.0
reflect1Display.PolarAxes.DeltaAngleMinor = 5.0
reflect1Display.PolarAxes.PolarAxisMajorTickSize = 0.0
reflect1Display.PolarAxes.PolarAxisTickRatioSize = 0.3
reflect1Display.PolarAxes.PolarAxisMajorTickThickness = 1.0
reflect1Display.PolarAxes.PolarAxisTickRatioThickness = 0.5
reflect1Display.PolarAxes.LastRadialAxisMajorTickSize = 0.0
reflect1Display.PolarAxes.LastRadialAxisTickRatioSize = 0.3
reflect1Display.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
reflect1Display.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
reflect1Display.PolarAxes.ArcMajorTickSize = 0.0
reflect1Display.PolarAxes.ArcTickRatioSize = 0.3
reflect1Display.PolarAxes.ArcMajorTickThickness = 1.0
reflect1Display.PolarAxes.ArcTickRatioThickness = 0.5
reflect1Display.PolarAxes.Use2DMode = 0
reflect1Display.PolarAxes.UseLogAxis = 0

# hide data in view
Hide(calculator1, renderView1)

# show color bar/color legend
reflect1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(threshold1)

# create a new 'Slice'
slice2 = Slice(Input=threshold1)
slice2.SliceType = 'Plane'
slice2.HyperTreeGridSlicer = 'Plane'
slice2.UseDual = 0
slice2.Crinkleslice = 0
slice2.Triangulatetheslice = 1
slice2.Mergeduplicatedpointsintheslice = 1
slice2.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice2.SliceType.Origin = [0.5, 0.875, 0.5]
slice2.SliceType.Normal = [1.0, 0.0, 0.0]
slice2.SliceType.Offset = 0.0

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice2.HyperTreeGridSlicer.Origin = [0.5, 0.875, 0.5]
slice2.HyperTreeGridSlicer.Normal = [1.0, 0.0, 0.0]
slice2.HyperTreeGridSlicer.Offset = 0.0

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice2.SliceType)

# Properties modified on slice2.SliceType
slice2.SliceType.Origin = [0.5, 0.875, 1.0]
slice2.SliceType.Normal = [0.0, 0.0, 1.0]

# show data in view
slice2Display = Show(slice2, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
slice2Display.Representation = 'Surface'
slice2Display.ColorArrayName = [None, '']
slice2Display.LookupTable = None
slice2Display.MapScalars = 1
slice2Display.MultiComponentsMapping = 0
slice2Display.InterpolateScalarsBeforeMapping = 1
slice2Display.Opacity = 1.0
slice2Display.PointSize = 2.0
slice2Display.LineWidth = 1.0
slice2Display.RenderLinesAsTubes = 0
slice2Display.RenderPointsAsSpheres = 0
slice2Display.Interpolation = 'Gouraud'
slice2Display.Specular = 0.0
slice2Display.SpecularColor = [1.0, 1.0, 1.0]
slice2Display.SpecularPower = 100.0
slice2Display.Luminosity = 0.0
slice2Display.Ambient = 0.0
slice2Display.Diffuse = 1.0
slice2Display.Roughness = 0.3
slice2Display.Metallic = 0.0
slice2Display.Texture = None
slice2Display.RepeatTextures = 1
slice2Display.InterpolateTextures = 0
slice2Display.SeamlessU = 0
slice2Display.SeamlessV = 0
slice2Display.UseMipmapTextures = 0
slice2Display.BaseColorTexture = None
slice2Display.NormalTexture = None
slice2Display.NormalScale = 1.0
slice2Display.MaterialTexture = None
slice2Display.OcclusionStrength = 1.0
slice2Display.EmissiveTexture = None
slice2Display.EmissiveFactor = [1.0, 1.0, 1.0]
slice2Display.FlipTextures = 0
slice2Display.BackfaceRepresentation = 'Follow Frontface'
slice2Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
slice2Display.BackfaceOpacity = 1.0
slice2Display.Position = [0.0, 0.0, 0.0]
slice2Display.Scale = [1.0, 1.0, 1.0]
slice2Display.Orientation = [0.0, 0.0, 0.0]
slice2Display.Origin = [0.0, 0.0, 0.0]
slice2Display.Pickable = 1
slice2Display.Triangulate = 0
slice2Display.UseShaderReplacements = 0
slice2Display.ShaderReplacements = ''
slice2Display.NonlinearSubdivisionLevel = 1
slice2Display.UseDataPartitions = 0
slice2Display.OSPRayUseScaleArray = 0
slice2Display.OSPRayScaleArray = ''
slice2Display.OSPRayScaleFunction = 'PiecewiseFunction'
slice2Display.OSPRayMaterial = 'None'
slice2Display.Orient = 0
slice2Display.OrientationMode = 'Direction'
slice2Display.SelectOrientationVectors = 'None'
slice2Display.Scaling = 0
slice2Display.ScaleMode = 'No Data Scaling Off'
slice2Display.ScaleFactor = 0.17500000000000002
slice2Display.SelectScaleArray = 'None'
slice2Display.GlyphType = 'Arrow'
slice2Display.UseGlyphTable = 0
slice2Display.GlyphTableIndexArray = 'None'
slice2Display.UseCompositeGlyphTable = 0
slice2Display.UseGlyphCullingAndLOD = 0
slice2Display.LODValues = []
slice2Display.ColorByLODIndex = 0
slice2Display.GaussianRadius = 0.00875
slice2Display.ShaderPreset = 'Sphere'
slice2Display.CustomTriangleScale = 3
slice2Display.CustomShader = """ // This custom shader code define a gaussian blur
 // Please take a look into vtkSMPointGaussianRepresentation.cxx
 // for other custom shader examples
 //VTK::Color::Impl
   float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
   float gaussian = exp(-0.5*dist2);
   opacity = opacity*gaussian;
"""
slice2Display.Emissive = 0
slice2Display.ScaleByArray = 0
slice2Display.SetScaleArray = [None, '']
slice2Display.ScaleArrayComponent = 0
slice2Display.UseScaleFunction = 1
slice2Display.ScaleTransferFunction = 'PiecewiseFunction'
slice2Display.OpacityByArray = 0
slice2Display.OpacityArray = [None, '']
slice2Display.OpacityArrayComponent = 0
slice2Display.OpacityTransferFunction = 'PiecewiseFunction'
slice2Display.DataAxesGrid = 'GridAxesRepresentation'
slice2Display.SelectionCellLabelBold = 0
slice2Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
slice2Display.SelectionCellLabelFontFamily = 'Arial'
slice2Display.SelectionCellLabelFontFile = ''
slice2Display.SelectionCellLabelFontSize = 18
slice2Display.SelectionCellLabelItalic = 0
slice2Display.SelectionCellLabelJustification = 'Left'
slice2Display.SelectionCellLabelOpacity = 1.0
slice2Display.SelectionCellLabelShadow = 0
slice2Display.SelectionPointLabelBold = 0
slice2Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
slice2Display.SelectionPointLabelFontFamily = 'Arial'
slice2Display.SelectionPointLabelFontFile = ''
slice2Display.SelectionPointLabelFontSize = 18
slice2Display.SelectionPointLabelItalic = 0
slice2Display.SelectionPointLabelJustification = 'Left'
slice2Display.SelectionPointLabelOpacity = 1.0
slice2Display.SelectionPointLabelShadow = 0
slice2Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
slice2Display.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
slice2Display.OSPRayScaleFunction.UseLogScale = 0

# init the 'Arrow' selected for 'GlyphType'
slice2Display.GlyphType.TipResolution = 6
slice2Display.GlyphType.TipRadius = 0.1
slice2Display.GlyphType.TipLength = 0.35
slice2Display.GlyphType.ShaftResolution = 6
slice2Display.GlyphType.ShaftRadius = 0.03
slice2Display.GlyphType.Invert = 0

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
slice2Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
slice2Display.ScaleTransferFunction.UseLogScale = 0

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
slice2Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
slice2Display.OpacityTransferFunction.UseLogScale = 0

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
slice2Display.DataAxesGrid.XTitle = 'X Axis'
slice2Display.DataAxesGrid.YTitle = 'Y Axis'
slice2Display.DataAxesGrid.ZTitle = 'Z Axis'
slice2Display.DataAxesGrid.XTitleFontFamily = 'Arial'
slice2Display.DataAxesGrid.XTitleFontFile = ''
slice2Display.DataAxesGrid.XTitleBold = 0
slice2Display.DataAxesGrid.XTitleItalic = 0
slice2Display.DataAxesGrid.XTitleFontSize = 12
slice2Display.DataAxesGrid.XTitleShadow = 0
slice2Display.DataAxesGrid.XTitleOpacity = 1.0
slice2Display.DataAxesGrid.YTitleFontFamily = 'Arial'
slice2Display.DataAxesGrid.YTitleFontFile = ''
slice2Display.DataAxesGrid.YTitleBold = 0
slice2Display.DataAxesGrid.YTitleItalic = 0
slice2Display.DataAxesGrid.YTitleFontSize = 12
slice2Display.DataAxesGrid.YTitleShadow = 0
slice2Display.DataAxesGrid.YTitleOpacity = 1.0
slice2Display.DataAxesGrid.ZTitleFontFamily = 'Arial'
slice2Display.DataAxesGrid.ZTitleFontFile = ''
slice2Display.DataAxesGrid.ZTitleBold = 0
slice2Display.DataAxesGrid.ZTitleItalic = 0
slice2Display.DataAxesGrid.ZTitleFontSize = 12
slice2Display.DataAxesGrid.ZTitleShadow = 0
slice2Display.DataAxesGrid.ZTitleOpacity = 1.0
slice2Display.DataAxesGrid.FacesToRender = 63
slice2Display.DataAxesGrid.CullBackface = 0
slice2Display.DataAxesGrid.CullFrontface = 1
slice2Display.DataAxesGrid.ShowGrid = 0
slice2Display.DataAxesGrid.ShowEdges = 1
slice2Display.DataAxesGrid.ShowTicks = 1
slice2Display.DataAxesGrid.LabelUniqueEdgesOnly = 1
slice2Display.DataAxesGrid.AxesToLabel = 63
slice2Display.DataAxesGrid.XLabelFontFamily = 'Arial'
slice2Display.DataAxesGrid.XLabelFontFile = ''
slice2Display.DataAxesGrid.XLabelBold = 0
slice2Display.DataAxesGrid.XLabelItalic = 0
slice2Display.DataAxesGrid.XLabelFontSize = 12
slice2Display.DataAxesGrid.XLabelShadow = 0
slice2Display.DataAxesGrid.XLabelOpacity = 1.0
slice2Display.DataAxesGrid.YLabelFontFamily = 'Arial'
slice2Display.DataAxesGrid.YLabelFontFile = ''
slice2Display.DataAxesGrid.YLabelBold = 0
slice2Display.DataAxesGrid.YLabelItalic = 0
slice2Display.DataAxesGrid.YLabelFontSize = 12
slice2Display.DataAxesGrid.YLabelShadow = 0
slice2Display.DataAxesGrid.YLabelOpacity = 1.0
slice2Display.DataAxesGrid.ZLabelFontFamily = 'Arial'
slice2Display.DataAxesGrid.ZLabelFontFile = ''
slice2Display.DataAxesGrid.ZLabelBold = 0
slice2Display.DataAxesGrid.ZLabelItalic = 0
slice2Display.DataAxesGrid.ZLabelFontSize = 12
slice2Display.DataAxesGrid.ZLabelShadow = 0
slice2Display.DataAxesGrid.ZLabelOpacity = 1.0
slice2Display.DataAxesGrid.XAxisNotation = 'Mixed'
slice2Display.DataAxesGrid.XAxisPrecision = 2
slice2Display.DataAxesGrid.XAxisUseCustomLabels = 0
slice2Display.DataAxesGrid.XAxisLabels = []
slice2Display.DataAxesGrid.YAxisNotation = 'Mixed'
slice2Display.DataAxesGrid.YAxisPrecision = 2
slice2Display.DataAxesGrid.YAxisUseCustomLabels = 0
slice2Display.DataAxesGrid.YAxisLabels = []
slice2Display.DataAxesGrid.ZAxisNotation = 'Mixed'
slice2Display.DataAxesGrid.ZAxisPrecision = 2
slice2Display.DataAxesGrid.ZAxisUseCustomLabels = 0
slice2Display.DataAxesGrid.ZAxisLabels = []
slice2Display.DataAxesGrid.UseCustomBounds = 0
slice2Display.DataAxesGrid.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
slice2Display.PolarAxes.Visibility = 0
slice2Display.PolarAxes.Translation = [0.0, 0.0, 0.0]
slice2Display.PolarAxes.Scale = [1.0, 1.0, 1.0]
slice2Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]
slice2Display.PolarAxes.EnableCustomBounds = [0, 0, 0]
slice2Display.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
slice2Display.PolarAxes.EnableCustomRange = 0
slice2Display.PolarAxes.CustomRange = [0.0, 1.0]
slice2Display.PolarAxes.PolarAxisVisibility = 1
slice2Display.PolarAxes.RadialAxesVisibility = 1
slice2Display.PolarAxes.DrawRadialGridlines = 1
slice2Display.PolarAxes.PolarArcsVisibility = 1
slice2Display.PolarAxes.DrawPolarArcsGridlines = 1
slice2Display.PolarAxes.NumberOfRadialAxes = 0
slice2Display.PolarAxes.AutoSubdividePolarAxis = 1
slice2Display.PolarAxes.NumberOfPolarAxis = 0
slice2Display.PolarAxes.MinimumRadius = 0.0
slice2Display.PolarAxes.MinimumAngle = 0.0
slice2Display.PolarAxes.MaximumAngle = 90.0
slice2Display.PolarAxes.RadialAxesOriginToPolarAxis = 1
slice2Display.PolarAxes.Ratio = 1.0
slice2Display.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
slice2Display.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
slice2Display.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
slice2Display.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
slice2Display.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
slice2Display.PolarAxes.PolarAxisTitleVisibility = 1
slice2Display.PolarAxes.PolarAxisTitle = 'Radial Distance'
slice2Display.PolarAxes.PolarAxisTitleLocation = 'Bottom'
slice2Display.PolarAxes.PolarLabelVisibility = 1
slice2Display.PolarAxes.PolarLabelFormat = '%-#6.3g'
slice2Display.PolarAxes.PolarLabelExponentLocation = 'Labels'
slice2Display.PolarAxes.RadialLabelVisibility = 1
slice2Display.PolarAxes.RadialLabelFormat = '%-#3.1f'
slice2Display.PolarAxes.RadialLabelLocation = 'Bottom'
slice2Display.PolarAxes.RadialUnitsVisibility = 1
slice2Display.PolarAxes.ScreenSize = 10.0
slice2Display.PolarAxes.PolarAxisTitleOpacity = 1.0
slice2Display.PolarAxes.PolarAxisTitleFontFamily = 'Arial'
slice2Display.PolarAxes.PolarAxisTitleFontFile = ''
slice2Display.PolarAxes.PolarAxisTitleBold = 0
slice2Display.PolarAxes.PolarAxisTitleItalic = 0
slice2Display.PolarAxes.PolarAxisTitleShadow = 0
slice2Display.PolarAxes.PolarAxisTitleFontSize = 12
slice2Display.PolarAxes.PolarAxisLabelOpacity = 1.0
slice2Display.PolarAxes.PolarAxisLabelFontFamily = 'Arial'
slice2Display.PolarAxes.PolarAxisLabelFontFile = ''
slice2Display.PolarAxes.PolarAxisLabelBold = 0
slice2Display.PolarAxes.PolarAxisLabelItalic = 0
slice2Display.PolarAxes.PolarAxisLabelShadow = 0
slice2Display.PolarAxes.PolarAxisLabelFontSize = 12
slice2Display.PolarAxes.LastRadialAxisTextOpacity = 1.0
slice2Display.PolarAxes.LastRadialAxisTextFontFamily = 'Arial'
slice2Display.PolarAxes.LastRadialAxisTextFontFile = ''
slice2Display.PolarAxes.LastRadialAxisTextBold = 0
slice2Display.PolarAxes.LastRadialAxisTextItalic = 0
slice2Display.PolarAxes.LastRadialAxisTextShadow = 0
slice2Display.PolarAxes.LastRadialAxisTextFontSize = 12
slice2Display.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
slice2Display.PolarAxes.SecondaryRadialAxesTextFontFamily = 'Arial'
slice2Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''
slice2Display.PolarAxes.SecondaryRadialAxesTextBold = 0
slice2Display.PolarAxes.SecondaryRadialAxesTextItalic = 0
slice2Display.PolarAxes.SecondaryRadialAxesTextShadow = 0
slice2Display.PolarAxes.SecondaryRadialAxesTextFontSize = 12
slice2Display.PolarAxes.EnableDistanceLOD = 1
slice2Display.PolarAxes.DistanceLODThreshold = 0.7
slice2Display.PolarAxes.EnableViewAngleLOD = 1
slice2Display.PolarAxes.ViewAngleLODThreshold = 0.7
slice2Display.PolarAxes.SmallestVisiblePolarAngle = 0.5
slice2Display.PolarAxes.PolarTicksVisibility = 1
slice2Display.PolarAxes.ArcTicksOriginToPolarAxis = 1
slice2Display.PolarAxes.TickLocation = 'Both'
slice2Display.PolarAxes.AxisTickVisibility = 1
slice2Display.PolarAxes.AxisMinorTickVisibility = 0
slice2Display.PolarAxes.ArcTickVisibility = 1
slice2Display.PolarAxes.ArcMinorTickVisibility = 0
slice2Display.PolarAxes.DeltaAngleMajor = 10.0
slice2Display.PolarAxes.DeltaAngleMinor = 5.0
slice2Display.PolarAxes.PolarAxisMajorTickSize = 0.0
slice2Display.PolarAxes.PolarAxisTickRatioSize = 0.3
slice2Display.PolarAxes.PolarAxisMajorTickThickness = 1.0
slice2Display.PolarAxes.PolarAxisTickRatioThickness = 0.5
slice2Display.PolarAxes.LastRadialAxisMajorTickSize = 0.0
slice2Display.PolarAxes.LastRadialAxisTickRatioSize = 0.3
slice2Display.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
slice2Display.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
slice2Display.PolarAxes.ArcMajorTickSize = 0.0
slice2Display.PolarAxes.ArcTickRatioSize = 0.3
slice2Display.PolarAxes.ArcMajorTickThickness = 1.0
slice2Display.PolarAxes.ArcTickRatioThickness = 0.5
slice2Display.PolarAxes.Use2DMode = 0
slice2Display.PolarAxes.UseLogAxis = 0

# hide data in view
Hide(threshold1, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(slice2Display, ('FIELD', 'vtkBlockColors'))

# show color bar/color legend
slice2Display.SetScalarBarVisibility(renderView1, True)

# create a new 'Calculator'
calculator2 = Calculator(Input=slice2)
calculator2.AttributeType = 'Cell Data'
calculator2.CoordinateResults = 0
calculator2.ResultNormals = 0
calculator2.ResultTCoords = 0
calculator2.ResultArrayName = 'Result'
calculator2.Function = ''
calculator2.ReplaceInvalidResults = 1
calculator2.ReplacementValue = 0.0
calculator2.ResultArrayType = 'Double'

# Properties modified on calculator2
calculator2.ResultArrayName = 'nE (1/cm3)'
calculator2.Function = 'n(E)'

# show data in view
calculator2Display = Show(calculator2, renderView1, 'GeometryRepresentation')

# get color transfer function/color map for 'nE1cm3'
nE1cm3LUT = GetColorTransferFunction('nE1cm3')

# trace defaults for the display properties.
calculator2Display.Representation = 'Surface'
calculator2Display.ColorArrayName = ['CELLS', 'nE (1/cm3)']
calculator2Display.LookupTable = nE1cm3LUT
calculator2Display.MapScalars = 1
calculator2Display.MultiComponentsMapping = 0
calculator2Display.InterpolateScalarsBeforeMapping = 1
calculator2Display.Opacity = 1.0
calculator2Display.PointSize = 2.0
calculator2Display.LineWidth = 1.0
calculator2Display.RenderLinesAsTubes = 0
calculator2Display.RenderPointsAsSpheres = 0
calculator2Display.Interpolation = 'Gouraud'
calculator2Display.Specular = 0.0
calculator2Display.SpecularColor = [1.0, 1.0, 1.0]
calculator2Display.SpecularPower = 100.0
calculator2Display.Luminosity = 0.0
calculator2Display.Ambient = 0.0
calculator2Display.Diffuse = 1.0
calculator2Display.Roughness = 0.3
calculator2Display.Metallic = 0.0
calculator2Display.Texture = None
calculator2Display.RepeatTextures = 1
calculator2Display.InterpolateTextures = 0
calculator2Display.SeamlessU = 0
calculator2Display.SeamlessV = 0
calculator2Display.UseMipmapTextures = 0
calculator2Display.BaseColorTexture = None
calculator2Display.NormalTexture = None
calculator2Display.NormalScale = 1.0
calculator2Display.MaterialTexture = None
calculator2Display.OcclusionStrength = 1.0
calculator2Display.EmissiveTexture = None
calculator2Display.EmissiveFactor = [1.0, 1.0, 1.0]
calculator2Display.FlipTextures = 0
calculator2Display.BackfaceRepresentation = 'Follow Frontface'
calculator2Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
calculator2Display.BackfaceOpacity = 1.0
calculator2Display.Position = [0.0, 0.0, 0.0]
calculator2Display.Scale = [1.0, 1.0, 1.0]
calculator2Display.Orientation = [0.0, 0.0, 0.0]
calculator2Display.Origin = [0.0, 0.0, 0.0]
calculator2Display.Pickable = 1
calculator2Display.Triangulate = 0
calculator2Display.UseShaderReplacements = 0
calculator2Display.ShaderReplacements = ''
calculator2Display.NonlinearSubdivisionLevel = 1
calculator2Display.UseDataPartitions = 0
calculator2Display.OSPRayUseScaleArray = 0
calculator2Display.OSPRayScaleArray = ''
calculator2Display.OSPRayScaleFunction = 'PiecewiseFunction'
calculator2Display.OSPRayMaterial = 'None'
calculator2Display.Orient = 0
calculator2Display.OrientationMode = 'Direction'
calculator2Display.SelectOrientationVectors = 'None'
calculator2Display.Scaling = 0
calculator2Display.ScaleMode = 'No Data Scaling Off'
calculator2Display.ScaleFactor = 0.17500000000000002
calculator2Display.SelectScaleArray = 'nE (1/cm3)'
calculator2Display.GlyphType = 'Arrow'
calculator2Display.UseGlyphTable = 0
calculator2Display.GlyphTableIndexArray = 'nE (1/cm3)'
calculator2Display.UseCompositeGlyphTable = 0
calculator2Display.UseGlyphCullingAndLOD = 0
calculator2Display.LODValues = []
calculator2Display.ColorByLODIndex = 0
calculator2Display.GaussianRadius = 0.00875
calculator2Display.ShaderPreset = 'Sphere'
calculator2Display.CustomTriangleScale = 3
calculator2Display.CustomShader = """ // This custom shader code define a gaussian blur
 // Please take a look into vtkSMPointGaussianRepresentation.cxx
 // for other custom shader examples
 //VTK::Color::Impl
   float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
   float gaussian = exp(-0.5*dist2);
   opacity = opacity*gaussian;
"""
calculator2Display.Emissive = 0
calculator2Display.ScaleByArray = 0
calculator2Display.SetScaleArray = [None, '']
calculator2Display.ScaleArrayComponent = 0
calculator2Display.UseScaleFunction = 1
calculator2Display.ScaleTransferFunction = 'PiecewiseFunction'
calculator2Display.OpacityByArray = 0
calculator2Display.OpacityArray = [None, '']
calculator2Display.OpacityArrayComponent = 0
calculator2Display.OpacityTransferFunction = 'PiecewiseFunction'
calculator2Display.DataAxesGrid = 'GridAxesRepresentation'
calculator2Display.SelectionCellLabelBold = 0
calculator2Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
calculator2Display.SelectionCellLabelFontFamily = 'Arial'
calculator2Display.SelectionCellLabelFontFile = ''
calculator2Display.SelectionCellLabelFontSize = 18
calculator2Display.SelectionCellLabelItalic = 0
calculator2Display.SelectionCellLabelJustification = 'Left'
calculator2Display.SelectionCellLabelOpacity = 1.0
calculator2Display.SelectionCellLabelShadow = 0
calculator2Display.SelectionPointLabelBold = 0
calculator2Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
calculator2Display.SelectionPointLabelFontFamily = 'Arial'
calculator2Display.SelectionPointLabelFontFile = ''
calculator2Display.SelectionPointLabelFontSize = 18
calculator2Display.SelectionPointLabelItalic = 0
calculator2Display.SelectionPointLabelJustification = 'Left'
calculator2Display.SelectionPointLabelOpacity = 1.0
calculator2Display.SelectionPointLabelShadow = 0
calculator2Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
calculator2Display.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
calculator2Display.OSPRayScaleFunction.UseLogScale = 0

# init the 'Arrow' selected for 'GlyphType'
calculator2Display.GlyphType.TipResolution = 6
calculator2Display.GlyphType.TipRadius = 0.1
calculator2Display.GlyphType.TipLength = 0.35
calculator2Display.GlyphType.ShaftResolution = 6
calculator2Display.GlyphType.ShaftRadius = 0.03
calculator2Display.GlyphType.Invert = 0

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
calculator2Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
calculator2Display.ScaleTransferFunction.UseLogScale = 0

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
calculator2Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
calculator2Display.OpacityTransferFunction.UseLogScale = 0

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
calculator2Display.DataAxesGrid.XTitle = 'X Axis'
calculator2Display.DataAxesGrid.YTitle = 'Y Axis'
calculator2Display.DataAxesGrid.ZTitle = 'Z Axis'
calculator2Display.DataAxesGrid.XTitleFontFamily = 'Arial'
calculator2Display.DataAxesGrid.XTitleFontFile = ''
calculator2Display.DataAxesGrid.XTitleBold = 0
calculator2Display.DataAxesGrid.XTitleItalic = 0
calculator2Display.DataAxesGrid.XTitleFontSize = 12
calculator2Display.DataAxesGrid.XTitleShadow = 0
calculator2Display.DataAxesGrid.XTitleOpacity = 1.0
calculator2Display.DataAxesGrid.YTitleFontFamily = 'Arial'
calculator2Display.DataAxesGrid.YTitleFontFile = ''
calculator2Display.DataAxesGrid.YTitleBold = 0
calculator2Display.DataAxesGrid.YTitleItalic = 0
calculator2Display.DataAxesGrid.YTitleFontSize = 12
calculator2Display.DataAxesGrid.YTitleShadow = 0
calculator2Display.DataAxesGrid.YTitleOpacity = 1.0
calculator2Display.DataAxesGrid.ZTitleFontFamily = 'Arial'
calculator2Display.DataAxesGrid.ZTitleFontFile = ''
calculator2Display.DataAxesGrid.ZTitleBold = 0
calculator2Display.DataAxesGrid.ZTitleItalic = 0
calculator2Display.DataAxesGrid.ZTitleFontSize = 12
calculator2Display.DataAxesGrid.ZTitleShadow = 0
calculator2Display.DataAxesGrid.ZTitleOpacity = 1.0
calculator2Display.DataAxesGrid.FacesToRender = 63
calculator2Display.DataAxesGrid.CullBackface = 0
calculator2Display.DataAxesGrid.CullFrontface = 1
calculator2Display.DataAxesGrid.ShowGrid = 0
calculator2Display.DataAxesGrid.ShowEdges = 1
calculator2Display.DataAxesGrid.ShowTicks = 1
calculator2Display.DataAxesGrid.LabelUniqueEdgesOnly = 1
calculator2Display.DataAxesGrid.AxesToLabel = 63
calculator2Display.DataAxesGrid.XLabelFontFamily = 'Arial'
calculator2Display.DataAxesGrid.XLabelFontFile = ''
calculator2Display.DataAxesGrid.XLabelBold = 0
calculator2Display.DataAxesGrid.XLabelItalic = 0
calculator2Display.DataAxesGrid.XLabelFontSize = 12
calculator2Display.DataAxesGrid.XLabelShadow = 0
calculator2Display.DataAxesGrid.XLabelOpacity = 1.0
calculator2Display.DataAxesGrid.YLabelFontFamily = 'Arial'
calculator2Display.DataAxesGrid.YLabelFontFile = ''
calculator2Display.DataAxesGrid.YLabelBold = 0
calculator2Display.DataAxesGrid.YLabelItalic = 0
calculator2Display.DataAxesGrid.YLabelFontSize = 12
calculator2Display.DataAxesGrid.YLabelShadow = 0
calculator2Display.DataAxesGrid.YLabelOpacity = 1.0
calculator2Display.DataAxesGrid.ZLabelFontFamily = 'Arial'
calculator2Display.DataAxesGrid.ZLabelFontFile = ''
calculator2Display.DataAxesGrid.ZLabelBold = 0
calculator2Display.DataAxesGrid.ZLabelItalic = 0
calculator2Display.DataAxesGrid.ZLabelFontSize = 12
calculator2Display.DataAxesGrid.ZLabelShadow = 0
calculator2Display.DataAxesGrid.ZLabelOpacity = 1.0
calculator2Display.DataAxesGrid.XAxisNotation = 'Mixed'
calculator2Display.DataAxesGrid.XAxisPrecision = 2
calculator2Display.DataAxesGrid.XAxisUseCustomLabels = 0
calculator2Display.DataAxesGrid.XAxisLabels = []
calculator2Display.DataAxesGrid.YAxisNotation = 'Mixed'
calculator2Display.DataAxesGrid.YAxisPrecision = 2
calculator2Display.DataAxesGrid.YAxisUseCustomLabels = 0
calculator2Display.DataAxesGrid.YAxisLabels = []
calculator2Display.DataAxesGrid.ZAxisNotation = 'Mixed'
calculator2Display.DataAxesGrid.ZAxisPrecision = 2
calculator2Display.DataAxesGrid.ZAxisUseCustomLabels = 0
calculator2Display.DataAxesGrid.ZAxisLabels = []
calculator2Display.DataAxesGrid.UseCustomBounds = 0
calculator2Display.DataAxesGrid.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
calculator2Display.PolarAxes.Visibility = 0
calculator2Display.PolarAxes.Translation = [0.0, 0.0, 0.0]
calculator2Display.PolarAxes.Scale = [1.0, 1.0, 1.0]
calculator2Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]
calculator2Display.PolarAxes.EnableCustomBounds = [0, 0, 0]
calculator2Display.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
calculator2Display.PolarAxes.EnableCustomRange = 0
calculator2Display.PolarAxes.CustomRange = [0.0, 1.0]
calculator2Display.PolarAxes.PolarAxisVisibility = 1
calculator2Display.PolarAxes.RadialAxesVisibility = 1
calculator2Display.PolarAxes.DrawRadialGridlines = 1
calculator2Display.PolarAxes.PolarArcsVisibility = 1
calculator2Display.PolarAxes.DrawPolarArcsGridlines = 1
calculator2Display.PolarAxes.NumberOfRadialAxes = 0
calculator2Display.PolarAxes.AutoSubdividePolarAxis = 1
calculator2Display.PolarAxes.NumberOfPolarAxis = 0
calculator2Display.PolarAxes.MinimumRadius = 0.0
calculator2Display.PolarAxes.MinimumAngle = 0.0
calculator2Display.PolarAxes.MaximumAngle = 90.0
calculator2Display.PolarAxes.RadialAxesOriginToPolarAxis = 1
calculator2Display.PolarAxes.Ratio = 1.0
calculator2Display.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
calculator2Display.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
calculator2Display.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
calculator2Display.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
calculator2Display.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
calculator2Display.PolarAxes.PolarAxisTitleVisibility = 1
calculator2Display.PolarAxes.PolarAxisTitle = 'Radial Distance'
calculator2Display.PolarAxes.PolarAxisTitleLocation = 'Bottom'
calculator2Display.PolarAxes.PolarLabelVisibility = 1
calculator2Display.PolarAxes.PolarLabelFormat = '%-#6.3g'
calculator2Display.PolarAxes.PolarLabelExponentLocation = 'Labels'
calculator2Display.PolarAxes.RadialLabelVisibility = 1
calculator2Display.PolarAxes.RadialLabelFormat = '%-#3.1f'
calculator2Display.PolarAxes.RadialLabelLocation = 'Bottom'
calculator2Display.PolarAxes.RadialUnitsVisibility = 1
calculator2Display.PolarAxes.ScreenSize = 10.0
calculator2Display.PolarAxes.PolarAxisTitleOpacity = 1.0
calculator2Display.PolarAxes.PolarAxisTitleFontFamily = 'Arial'
calculator2Display.PolarAxes.PolarAxisTitleFontFile = ''
calculator2Display.PolarAxes.PolarAxisTitleBold = 0
calculator2Display.PolarAxes.PolarAxisTitleItalic = 0
calculator2Display.PolarAxes.PolarAxisTitleShadow = 0
calculator2Display.PolarAxes.PolarAxisTitleFontSize = 12
calculator2Display.PolarAxes.PolarAxisLabelOpacity = 1.0
calculator2Display.PolarAxes.PolarAxisLabelFontFamily = 'Arial'
calculator2Display.PolarAxes.PolarAxisLabelFontFile = ''
calculator2Display.PolarAxes.PolarAxisLabelBold = 0
calculator2Display.PolarAxes.PolarAxisLabelItalic = 0
calculator2Display.PolarAxes.PolarAxisLabelShadow = 0
calculator2Display.PolarAxes.PolarAxisLabelFontSize = 12
calculator2Display.PolarAxes.LastRadialAxisTextOpacity = 1.0
calculator2Display.PolarAxes.LastRadialAxisTextFontFamily = 'Arial'
calculator2Display.PolarAxes.LastRadialAxisTextFontFile = ''
calculator2Display.PolarAxes.LastRadialAxisTextBold = 0
calculator2Display.PolarAxes.LastRadialAxisTextItalic = 0
calculator2Display.PolarAxes.LastRadialAxisTextShadow = 0
calculator2Display.PolarAxes.LastRadialAxisTextFontSize = 12
calculator2Display.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
calculator2Display.PolarAxes.SecondaryRadialAxesTextFontFamily = 'Arial'
calculator2Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''
calculator2Display.PolarAxes.SecondaryRadialAxesTextBold = 0
calculator2Display.PolarAxes.SecondaryRadialAxesTextItalic = 0
calculator2Display.PolarAxes.SecondaryRadialAxesTextShadow = 0
calculator2Display.PolarAxes.SecondaryRadialAxesTextFontSize = 12
calculator2Display.PolarAxes.EnableDistanceLOD = 1
calculator2Display.PolarAxes.DistanceLODThreshold = 0.7
calculator2Display.PolarAxes.EnableViewAngleLOD = 1
calculator2Display.PolarAxes.ViewAngleLODThreshold = 0.7
calculator2Display.PolarAxes.SmallestVisiblePolarAngle = 0.5
calculator2Display.PolarAxes.PolarTicksVisibility = 1
calculator2Display.PolarAxes.ArcTicksOriginToPolarAxis = 1
calculator2Display.PolarAxes.TickLocation = 'Both'
calculator2Display.PolarAxes.AxisTickVisibility = 1
calculator2Display.PolarAxes.AxisMinorTickVisibility = 0
calculator2Display.PolarAxes.ArcTickVisibility = 1
calculator2Display.PolarAxes.ArcMinorTickVisibility = 0
calculator2Display.PolarAxes.DeltaAngleMajor = 10.0
calculator2Display.PolarAxes.DeltaAngleMinor = 5.0
calculator2Display.PolarAxes.PolarAxisMajorTickSize = 0.0
calculator2Display.PolarAxes.PolarAxisTickRatioSize = 0.3
calculator2Display.PolarAxes.PolarAxisMajorTickThickness = 1.0
calculator2Display.PolarAxes.PolarAxisTickRatioThickness = 0.5
calculator2Display.PolarAxes.LastRadialAxisMajorTickSize = 0.0
calculator2Display.PolarAxes.LastRadialAxisTickRatioSize = 0.3
calculator2Display.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
calculator2Display.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
calculator2Display.PolarAxes.ArcMajorTickSize = 0.0
calculator2Display.PolarAxes.ArcTickRatioSize = 0.3
calculator2Display.PolarAxes.ArcMajorTickThickness = 1.0
calculator2Display.PolarAxes.ArcTickRatioThickness = 0.5
calculator2Display.PolarAxes.Use2DMode = 0
calculator2Display.PolarAxes.UseLogAxis = 0

# hide data in view
Hide(slice2, renderView1)

# show color bar/color legend
calculator2Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get opacity transfer function/opacity map for 'nE1cm3'
nE1cm3PWF = GetOpacityTransferFunction('nE1cm3')

# convert to log space
nE1cm3LUT.MapControlPointsToLogSpace()

# Properties modified on nE1cm3LUT
# NOTE: Use log scale for n(E)
nE1cm3LUT.UseLogScale = 1

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
# NOTE: Change n(E) color palette here
# See https://www.paraview.org/Wiki/images/7/73/Luts.png for examples of color palettes
nE1cm3LUT.ApplyPreset('Black-Body Radiation', True)

# get color legend/bar for nE1cm3LUT in view renderView1
nE1cm3LUTColorBar = GetScalarBar(nE1cm3LUT, renderView1)

# Properties modified on nE1cm3LUTColorBar
# NOTE: [0.0, 0.0, 0.0] for black, [1.0, 1.0, 1.0] for white
nE1cm3LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
nE1cm3LUTColorBar.LabelColor = [0.0, 0.0, 0.0]

# set active source
SetActiveSource(reflect1)

# Rescale transfer function
# NOTE: Set color scale min/max values for E/N
eNTdLUT.RescaleTransferFunction(0.0, 1200.0)

# Rescale transfer function
# NOTE: Set color scale min/max values for E/N
eNTdPWF.RescaleTransferFunction(0.0, 1200.0)

# set active source
SetActiveSource(calculator2)

# Rescale transfer function
# NOTE: Set color scale min/max values for n(E)
nE1cm3LUT.RescaleTransferFunction(100000000.0, 1e+15)

# Rescale transfer function
# NOTE: Set color scale min/max values for n(E)
nE1cm3PWF.RescaleTransferFunction(100000000.0, 1e+15)

# reset view to fit data
renderView1.ResetCamera()

# NOTE: turn off little corner x/y/z axis figure
renderView1.OrientationAxesVisibility = 0

# change scalar bar placement
nE1cm3LUTColorBar.WindowLocation = 'AnyLocation'
# NOTE: Change n(E) colorbar placement
nE1cm3LUTColorBar.Position = [0.0, 0.57]
# NOTE: Change n(E) colorbar length
nE1cm3LUTColorBar.ScalarBarLength = 0.35000000000000001

# change scalar bar placement
eNTdLUTColorBar.WindowLocation = 'AnyLocation'
# NOTE: Change E/N colorbar placement
eNTdLUTColorBar.Position = [0.0, 0.15]
# NOTE: Change E/N colorbar length
eNTdLUTColorBar.ScalarBarLength = 0.3500000000000001

#NOTE: colorbar font can be set using local ttf file
nE1cm3LUTColorBar.TitleFontFamily = 'File'
nE1cm3LUTColorBar.TitleFontFile = '/home1/04361/ndeak/fonts/cmunrm.ttf'
nE1cm3LUTColorBar.LabelFontFamily = 'File'
nE1cm3LUTColorBar.LabelFontFile = '/home1/04361/ndeak/fonts/cmunrm.ttf'
eNTdLUTColorBar.TitleFontFamily = 'File'
eNTdLUTColorBar.TitleFontFile = '/home1/04361/ndeak/fonts/cmunrm.ttf'
eNTdLUTColorBar.LabelFontFamily = 'File'
eNTdLUTColorBar.LabelFontFile = '/home1/04361/ndeak/fonts/cmunrm.ttf'

# current camera placement for renderView1
renderView1.CameraPosition = [1.0, 0.875, 6.133466558764061]
renderView1.CameraFocalPoint = [1.0, 0.875, 0.9995000064373016]
renderView1.CameraParallelScale = 1.3287683206614924

# save screenshot
# NOTE: Set file name/location here, make sure resolution matches one at top
# SaveScreenshot('/home1/04361/ndeak/bourdon_2p5mm_50um_nE_EN_colorscale.png', renderView1, ImageResolution=[2048, 1792],
SaveScreenshot('/home1/04361/ndeak/PAC_7lev_noCap_nE_EN_colorscale.png', renderView1, ImageResolution=[1024, 896],
    FontScaling='Scale fonts proportionally',
    # FontScaling='Do not scale fonts',
    OverrideColorPalette='',
    StereoMode='No change',
    # NOTE gray background = 0, transparent = 1
    TransparentBackground=1, 
    # PNG options
    CompressionLevel='5')

# change scalar bar placement
nE1cm3LUTColorBar.Position = [1.05, 0.6907407407407407]

# change scalar bar placement
eNTdLUTColorBar.Position = [1.05, 0.14259259259259255]

# current camera placement for renderView1
# NOTE: Set x and y center coordinates for image (Position and Focal point should match?)
# NOTE: z parameter for Position should be sightly greater than FocalPoint, or image wont appear
renderView1.CameraPosition = [1.0, 0.25, 1.01]
renderView1.CameraFocalPoint = [1.0, 0.25, 1.0]
# NOTE: controls zoom (lower value -> more zoomed in, seems to range from 0->180)
renderView1.CameraViewAngle = 174.0
# NOTE: does not seem to change image at all...
renderView1.CameraParallelScale = 1.3287683206614924

# save screenshot
# NOTE: Set file name/location here, make sure resolution matches one at top
SaveScreenshot('/home1/04361/ndeak/PAC_7lev_noCap_nE_EN.png', renderView1, ImageResolution=[2048, 1792],
    FontScaling='Scale fonts proportionally',
    OverrideColorPalette='',
    StereoMode='No change',
    # NOTE gray background = 0, transparent = 1
    TransparentBackground=0, 
    # PNG options
    CompressionLevel='5')

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.CameraPosition = [1.0, 0.875, 6.133466558764061]
renderView1.CameraFocalPoint = [1.0739709821253483, 0.875, 1.3246191287031719]
renderView1.CameraViewAngle = 5.438388625592417
renderView1.CameraParallelScale = 1.3287683206614924

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
