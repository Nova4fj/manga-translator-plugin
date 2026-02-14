# UI Design: Manga Translator Plugin

## 🎨 Design Philosophy

### Core Principles
- **GIMP Native**: Seamless integration with GIMP's existing UI paradigms and visual language
- **Progressive Disclosure**: Simple defaults with advanced options available when needed
- **Visual Feedback**: Clear progress indication and real-time preview capabilities
- **Workflow Flexibility**: Support for different user skill levels and translation approaches
- **Context Awareness**: UI adapts based on detected content and user preferences

### Design Goals
1. **Minimize Cognitive Load**: Reduce decision fatigue through smart defaults
2. **Maintain Creative Flow**: Quick access to common operations without disrupting artistic workflow
3. **Error Prevention**: Clear validation and confirmation for destructive operations
4. **Accessibility**: Support for keyboard navigation and screen readers
5. **Responsiveness**: Efficient UI updates during long-running operations

## 🔗 GIMP Integration Points

### Menu Integration

#### Primary Menu Location
```
Filters → Manga Translator → [Submenu Items]
```

**Submenu Structure**:
```
Filters
├── Manga Translator
│   ├── Auto Translate Page          [Ctrl+Shift+T]
│   ├── Semi-Auto Translate...       [Ctrl+Alt+T]  
│   ├── Manual Translation Tools     [Ctrl+M]
│   ├── ─────────────────────
│   ├── Batch Process...
│   ├── Translation Memory...
│   ├── ─────────────────────
│   ├── Settings...                  [Ctrl+,]
│   └── Help & Documentation
```

**Alternative Access Points**:
- **Context Menu**: Right-click on image layer → "Manga Translator"
- **Toolbar**: Custom toolbar button (user-configurable)
- **Dockable Dialog**: Persistent panel for power users

#### Menu Item Behaviors

**Auto Translate Page**:
- Immediate execution with progress dialog
- Uses saved preferences for all settings
- Single-click solution for standard manga pages

**Semi-Auto Translate...**:
- Opens bubble detection preview dialog
- Allows user review and adjustment before processing
- Balanced control vs. speed approach

**Manual Translation Tools**:
- Activates selection tools for text regions
- Opens floating tool palette for manual workflow
- Maximum user control for complex layouts

### Keyboard Shortcuts

**Global Shortcuts**:
- `Ctrl+Shift+T`: Auto translate current page
- `Ctrl+Alt+T`: Semi-auto translate with preview
- `Ctrl+M`: Manual translation tools
- `Ctrl+Shift+B`: Batch process multiple images
- `Ctrl+,`: Open settings dialog

**Modal Shortcuts** (within translation dialogs):
- `Space`: Toggle preview overlay
- `Enter`: Accept current step/translation
- `Escape`: Cancel current operation
- `Tab`: Cycle through text regions
- `F1`: Context-sensitive help

## 📱 Dialog Design System

### Visual Hierarchy

**Color Scheme** (GIMP-compliant):
- **Primary Actions**: GIMP blue (#0078d4)
- **Warning States**: Orange (#ff8800)  
- **Error States**: Red (#d13438)
- **Success States**: Green (#107c10)
- **Background**: GIMP theme background
- **Text**: GIMP theme foreground

**Typography**:
- **Headers**: GIMP default UI font, bold, 12pt
- **Body Text**: GIMP default UI font, regular, 10pt
- **Captions**: GIMP default UI font, regular, 9pt
- **Code/Paths**: Monospace, 9pt

**Spacing System**:
- **Micro**: 4px (element padding)
- **Small**: 8px (related elements)
- **Medium**: 16px (section separation)
- **Large**: 24px (major section breaks)
- **XLarge**: 32px (dialog margins)

### Main Translation Dialog

#### Layout Structure
```
┌─────────────────────────────────────────────────────────────────────────┐
│ Manga Translator - [filename.jpg]                               [?][X] │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐  ┌──────────────────────────────────────────────┐ │
│  │   Mode Select   │  │               Preview Area                   │ │
│  │                 │  │                                              │ │
│  │ ○ Auto          │  │    [Original Image with Overlay]            │ │
│  │ ● Semi-Auto     │  │                                              │ │
│  │ ○ Manual        │  │    ┌──────────────────────────────────────┐ │ │
│  │                 │  │    │  Toggle: ☑ Show Bubbles             │ │ │
│  │ [Settings...] │  │    │          ☑ Show Text                │ │ │
│  │                 │  │    │          ☑ Show Translations       │ │ │
│  │ Languages:      │  │    └──────────────────────────────────────┘ │ │
│  │ From: [Auto ▼]  │  │                                              │ │
│  │ To:   [EN   ▼]  │  │    Zoom: [Fit][100%][200%]    [⟲][⟳]     │ │
│  │                 │  └──────────────────────────────────────────────┘ │
│  │ Translation:    │                                                   │
│  │ Engine: [GPT▼]  │  ┌──────────────────────────────────────────────┐ │
│  │                 │  │             Processing Steps                 │ │
│  │ Font Options:   │  │                                              │ │
│  │ Size: [Auto▼]   │  │  1. ✓ Image Analysis     (0.2s)            │ │
│  │ Style:[Comic▼]  │  │  2. ⟳ Bubble Detection   (1.3s)            │ │
│  │                 │  │  3. ⏸ Text Extraction    (2.1s)            │ │
│  │                 │  │  4. ⏸ Translation        (----)            │ │
│  │                 │  │  5. ⏸ Text Removal       (----)            │ │
│  │                 │  │  6. ⏸ Typesetting        (----)            │ │
│  └─────────────────┘  │                                              │ │
│                       │  Overall Progress: ▓▓▓▓░░░░░░ 34%          │ │
│                       │                                              │ │
│                       │  [Pause] [Cancel]           [< Back][Next >] │ │
│                       └──────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────┤
│                                    [Help] [Apply] [Cancel] [OK]      │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Responsive Behavior

**Large Dialogs (1200px+ width)**:
- Side-by-side layout with preview and controls
- Full-size image preview with zoom controls
- Expanded settings panels

**Medium Dialogs (800-1200px width)**:
- Stacked layout with tabbed controls
- Scrollable preview area
- Collapsed advanced settings

**Small Dialogs (<800px width)**:
- Wizard-style step progression
- Minimal preview thumbnails
- Essential controls only

### Semi-Auto Mode Dialog

#### Bubble Detection Review Interface
```
┌─────────────────────────────────────────────────────────────────────────┐
│ Review Detected Bubbles - Step 1 of 6                           [?][X] │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Original Image                                │   │
│  │                                                                  │   │
│  │     ┌─────────────┐         ┌──────────────────┐                │   │
│  │     │   Bubble 1  │  ←─●    │     Bubble 2     │  ←─●           │   │
│  │     │ (Detected)  │  95%    │   (Detected)     │  87%           │   │
│  │     └─────────────┘         └──────────────────┘                │   │
│  │                                                                  │   │
│  │              ┌────────────────────────┐                         │   │
│  │              │      Bubble 3          │  ←─●                    │   │
│  │              │    (Detected)          │  92%                    │   │
│  │              └────────────────────────┘                         │   │
│  │                                                                  │   │
│  │  ┌─ Manual Bubble (Click and drag to create) ─┐                │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─ Detection Controls ────────────────────────────────────────────────┐ │
│  │                                                                     │ │
│  │ Sensitivity: ▓▓▓▓▓▓▓░░░ 75%        Min Size: [500px²]             │ │
│  │                                                                     │ │
│  │ Bubble Types: ☑ Speech  ☑ Thought  ☑ Narration  ☐ SFX           │ │
│  │                                                                     │ │
│  │ Actions:                                                            │ │
│  │ [Re-detect] [Add Manual] [Delete Selected] [Adjust Boundaries]     │ │
│  │                                                                     │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─ Detected Bubbles List ─────────────────────────────────────────────┐ │
│  │ # │ Type     │ Size    │ Confidence │ Text Preview     │ Actions    │ │
│  │───┼──────────┼─────────┼────────────┼──────────────────┼────────────│ │
│  │ 1 │ Speech   │ 45×23   │    95%     │ "こんにちは..."   │ [Edit][Del]│ │
│  │ 2 │ Speech   │ 67×34   │    87%     │ "そうですね"     │ [Edit][Del]│ │
│  │ 3 │ Thought  │ 89×12   │    92%     │ "どうしよう..."   │ [Edit][Del]│ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  ☑ Remember these detection settings                                   │
│                              [< Back] [Skip to Manual] [Continue >]    │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Text Review Interface
```
┌─────────────────────────────────────────────────────────────────────────┐
│ Review Extracted Text - Step 2 of 6                             [?][X] │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─ Current Bubble: 1 of 3 ──┬── Navigation ──────────────────────────┐ │
│  │                           │                                         │ │
│  │   ┌─────────────────────┐ │  [◄ Prev] [Next ►] [Jump to: [1▼]]    │ │
│  │   │   Cropped Region    │ │                                         │ │
│  │   │                     │ │  Confidence: ████████░ 87%             │ │
│  │   │  こんにちは、        │ │                                         │ │
│  │   │  お元気ですか？      │ │  Auto-detected: Japanese                │ │
│  │   │                     │ │                                         │ │
│  │   └─────────────────────┘ │  Language: [Japanese ▼]               │ │
│  └───────────────────────────┴─────────────────────────────────────────┘ │
│                                                                         │
│  ┌─ Extracted Text ────────────────────────────────────────────────────┐ │
│  │ Original (Japanese):                                                │ │
│  │ ┌─────────────────────────────────────────────────────────────────┐ │ │
│  │ │ こんにちは、お元気ですか？                                        │ │ │
│  │ └─────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                     │ │
│  │ Romanization:                                                       │ │
│  │ ┌─────────────────────────────────────────────────────────────────┐ │ │
│  │ │ Konnichiwa, ogenki desu ka?                                     │ │ │
│  │ └─────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                     │ │
│  │ ☑ Text is correct  ☐ Needs correction                            │ │
│  │                                                                     │ │
│  │ Actions: [Re-OCR] [Manual Entry] [Skip This Bubble]               │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─ Batch Actions ─────────────────────────────────────────────────────┐ │
│  │ [Accept All High-Confidence] [Re-OCR All] [Manual Review Mode]     │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Progress: Bubble 1 of 3 ▓▓▓▓▓▓▓▓▓░ 90%                              │
│                              [< Back] [Skip All] [Continue >]          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Manual Mode Interface

#### Selection Tools Palette
```
┌─────────────────────────────────────────────┐
│ Manga Translation Tools              [×]    │
├─────────────────────────────────────────────┤
│                                             │
│  Selection Tools:                          │
│  ┌──┐ ┌──┐ ┌──┐ ┌──┐                      │
│  │🔲│ │○ │ │✂ │ │🖊│                      │
│  └──┘ └──┘ └──┘ └──┘                      │
│  Rect Oval Lasso Bezier                    │
│                                             │
│  Text Region Actions:                      │
│  [Extract Text] [Translate] [Clear]        │
│                                             │
│  ┌─ Current Selection ─────────────────────┐ │
│  │ Type: Rectangle                        │ │
│  │ Size: 150 × 45 px                     │ │
│  │ Confidence: --                         │ │
│  │ Status: Ready for OCR                  │ │
│  └────────────────────────────────────────┘ │
│                                             │
│  ┌─ Quick Settings ────────────────────────┐ │
│  │ OCR: [manga-ocr ▼]                    │ │
│  │ Lang: [Japanese ▼]                    │ │
│  │ Trans: [OpenAI ▼]                     │ │
│  │ Font: [Auto ▼]                        │ │
│  └────────────────────────────────────────┘ │
│                                             │
│  [Process Selection] [Batch Mode]          │
│                                             │
└─────────────────────────────────────────────┘
```

### Settings Dialog

#### Tabbed Settings Interface
```
┌─────────────────────────────────────────────────────────────────────────┐
│ Manga Translator Settings                                        [?][X] │
├─┬───────┬───────────┬──────────┬───────────┬──────────────────────────────┤
│ │General│Translation│Typography│Performance│Advanced                      │
├─┴───────┴───────────┴──────────┴───────────┴──────────────────────────────┤
│                                                                         │
│ ┌─ Language Settings ─────────────────────────────────────────────────────┐ │
│ │                                                                        │ │
│ │ Default Source Language: [Auto-detect        ▼]                      │ │
│ │ Default Target Language: [English (US)       ▼]                      │ │
│ │                                                                        │ │
│ │ ☑ Auto-detect source language                                        │ │
│ │ ☑ Show language confidence scores                                     │ │
│ │ ☐ Warn when switching between language pairs                         │ │
│ └────────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│ ┌─ Default Workflow ──────────────────────────────────────────────────────┐ │
│ │                                                                        │ │
│ │ Preferred Mode: ● Auto  ○ Semi-Auto  ○ Manual                        │ │
│ │                                                                        │ │
│ │ Auto Mode Behavior:                                                    │ │
│ │ ☑ Show progress dialog                                                │ │
│ │ ☑ Create backup layer                                                 │ │
│ │ ☐ Auto-save result                                                    │ │
│ │ ☑ Group translation layers                                            │ │
│ └────────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│ ┌─ File Handling ─────────────────────────────────────────────────────────┐ │
│ │                                                                        │ │
│ │ Temporary Files Location:                                              │ │
│ │ [/tmp/manga-translator          ] [Browse...] [Clear Cache]           │ │
│ │                                                                        │ │
│ │ Model Storage Location:                                                │ │
│ │ [~/.gimp-2.10/manga-translator  ] [Browse...] [Update Models]        │ │
│ │                                                                        │ │
│ │ ☑ Clean temporary files on exit                                      │ │
│ │ ☐ Compress saved translation projects                                 │ │
│ │ ☐ Auto-backup translation memories                                    │ │
│ └────────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                 [Restore Defaults] [Help] [Cancel] [OK]                │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Translation Tab
```
┌─ Translation Engines ───────────────────────────────────────────────────────┐
│                                                                            │
│ Primary Engine: [OpenAI GPT-4 ▼]                                          │
│ Fallback Order: [1. DeepL] [2. Google] [3. Argos] [Reorder...]           │
│                                                                            │
│ ┌─ Engine Configuration ──────────────────────────────────────────────────┐  │
│ │                                                                         │  │
│ │ OpenAI Settings:                                                        │  │
│ │ API Key: [••••••••••••••••••••••••••••••••] [Test] [Change]           │  │
│ │ Model: [gpt-4 ▼]                                                       │  │
│ │ Temperature: ▓▓▓░░░░░░░ 0.3 (Creative ← → Conservative)               │  │
│ │                                                                         │  │
│ │ DeepL Settings:                                                         │  │
│ │ Auth Key: [••••••••••••••••••••••••••••••••] [Test] [Change]          │  │
│ │ Formality: [Less formal ▼] (Better for dialogue)                      │  │
│ │                                                                         │  │
│ │ Argos Translate (Offline):                                              │  │
│ │ Status: ✓ Installed    Models: 12 language pairs                      │  │
│ │ [Download More Models] [Update Existing]                               │  │
│ │                                                                         │  │
│ └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│ ┌─ Translation Quality ───────────────────────────────────────────────────┐  │
│ │                                                                         │  │
│ │ Minimum Confidence: ▓▓▓▓▓▓▓░░░ 70%                                    │  │
│ │ Retry Failed Translations: ☑ Up to 3 times                            │  │
│ │ Request Alternative Translations: ☑ When confidence < 80%              │  │
│ │                                                                         │  │
│ │ Context Settings:                                                       │  │
│ │ ☑ Use previous bubble context                                          │  │
│ │ ☑ Maintain character speech patterns                                   │  │
│ │ ☐ Include scene descriptions (experimental)                            │  │
│ │                                                                         │  │
│ └─────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Workflow Modes Interface Design

### Auto Mode: One-Click Experience

**Interaction Flow**:
1. User clicks "Auto Translate Page" from menu
2. Brief settings confirmation dialog (3 seconds, auto-dismiss)
3. Processing dialog with live progress updates
4. Results preview with accept/reject options
5. Final layer creation in GIMP

**Progress Dialog Design**:
```
┌─────────────────────────────────────────────────────────┐
│ Auto Translation Progress                        [✕]    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Translating: [manga_page_01.jpg]                      │
│                                                         │
│  Current Step: Extracting text from bubble 3 of 7      │
│                                                         │
│  ╔═══════════════════════════════════════════════════╗  │
│  ║▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░║  │
│  ╚═══════════════════════════════════════════════════╝  │
│                    Progress: 67%                       │
│                                                         │
│  Estimated time remaining: 12 seconds                  │
│                                                         │
│  ┌─ Processing Log ────────────────────────────────────┐ │
│  │ ✓ Image analysis complete (0.3s)                   │ │
│  │ ✓ Found 7 speech bubbles (1.2s)                    │ │
│  │ ✓ Extracted bubble 1: "こんにちは"                   │ │
│  │ ✓ Extracted bubble 2: "元気？"                      │ │
│  │ ⏳ Extracting bubble 3...                          │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                         │
│               [Show Details] [Pause] [Cancel]          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Semi-Auto Mode: Guided Workflow

**Step-by-Step Interface**:

**Step 1: Bubble Detection Review**
- Visual overlay showing detected bubbles
- Confidence indicators for each detection
- Tools to adjust boundaries or add missed bubbles
- Batch actions for common adjustments

**Step 2: Text Extraction Review**
- Per-bubble text preview and editing
- OCR confidence display
- Language detection confirmation
- Manual text entry for low-confidence extractions

**Step 3: Translation Review**
- Side-by-side original and translated text
- Alternative translation options
- Context adjustment tools
- Terminology dictionary integration

**Step 4: Typesetting Preview**
- Real-time preview of text placement
- Font and sizing adjustments
- Layout modification tools
- Style matching options

**Navigation Controls**:
```
┌──────────────────────────────────────────────────────────────┐
│ [◄◄ First] [◄ Previous] Step 2 of 4 [Next ►] [Last ►►]     │
│                                                              │
│ Quick Actions: [Skip This] [Apply to All] [Save & Continue] │
└──────────────────────────────────────────────────────────────┘
```

### Manual Mode: Expert Control

**Floating Tool Palette Design**:
- Minimal, non-intrusive interface
- Quick access to essential tools
- Context-sensitive options
- Customizable tool arrangement

**Selection Workflow**:
1. User selects text region with GIMP tools
2. Floating palette shows region analysis
3. One-click OCR extraction
4. Inline translation editing
5. Direct text placement controls

## 📊 Progress Indicators and Feedback

### Progress Visualization System

**Micro-Progress** (Individual Operations):
- Spinner animations for quick tasks (<3 seconds)
- Determinate progress bars for predictable tasks
- Pulse animation for indeterminate tasks
- Color coding: Blue (processing), Green (success), Red (error)

**Macro-Progress** (Overall Workflow):
- Multi-stage progress indicators
- Step completion visualization
- Time estimates and remaining duration
- Parallel processing indicators for independent tasks

**Real-Time Updates**:
```
Processing Animation Examples:

Bubble Detection:    ⟳ Analyzing regions... (42% complete)
OCR Extraction:      📖 Reading text bubble 3 of 7...
Translation:         🌐 Translating via OpenAI... (Queue: 2)
Text Removal:        🖌️ Inpainting region 4 of 7... (67%)
Typesetting:         ✍️ Rendering English text...
Finalizing:          ✅ Creating GIMP layers...
```

### Error Handling and Recovery

**Error Dialog Design**:
```
┌─────────────────────────────────────────────────────────┐
│ ⚠️ Translation Warning                           [!][✕] │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Issue: Low OCR confidence for bubble 3                 │
│ Confidence: 23% (threshold: 70%)                       │
│                                                         │
│ Extracted text: "�できます???"                           │
│                                                         │
│ Suggested actions:                                      │
│ ● Try different OCR engine                             │
│ ○ Manually enter correct text                          │
│ ○ Skip this bubble for now                             │
│ ○ Lower confidence threshold                           │
│                                                         │
│ ☑ Apply choice to similar low-confidence bubbles       │
│                                                         │
│            [Try Again] [Manual Entry] [Skip]           │
└─────────────────────────────────────────────────────────┘
```

**Recovery Options Hierarchy**:
1. **Automatic Recovery**: Silent fallback to alternative methods
2. **Guided Recovery**: Suggest specific actions with one-click fixes
3. **Manual Recovery**: Provide tools for user intervention
4. **Graceful Degradation**: Continue with partial results

## 🔍 Preview and Undo Support

### Live Preview System

**Preview Modes**:
- **Overlay Mode**: Transparent overlay showing translations
- **Side-by-Side**: Original and translated versions
- **Before/After**: Toggle between states
- **Diff Mode**: Highlight changed regions

**Preview Controls**:
```
┌─ Preview Options ──────────────────────────────────────────┐
│                                                            │
│ Display: ● Overlay ○ Side-by-Side ○ Before/After         │
│                                                            │
│ Show: ☑ Bubbles ☑ Original Text ☑ Translations           │
│                                                            │
│ Opacity: ▓▓▓▓▓▓▓▓░░ 80%                                  │
│                                                            │
│ [Update Preview] [Full Screen] [Save Preview Image]       │
└────────────────────────────────────────────────────────────┘
```

### Undo/Redo Integration

**GIMP Undo System Integration**:
- Each translation step creates undo point
- Grouped operations for complex workflows
- Persistent undo across plugin sessions
- Custom undo descriptions for clarity

**Translation-Specific Undo Actions**:
```
Undo History:
├── Original Image
├── Add Bubble Detection Layer
├── Create Text Extraction Layer
├── Remove Original Text (Bubble 1)
├── Remove Original Text (Bubble 2)
├── Add English Text (Bubble 1)
├── Add English Text (Bubble 2)
└── Merge Translation Layers ← Current State
```

**Multi-Level Undo Controls**:
- **Step Undo**: Undo last operation within current workflow
- **Bubble Undo**: Undo all changes to specific bubble
- **Full Undo**: Return to pre-translation state
- **Selective Undo**: Choose specific operations to undo

## 📱 Responsive Design Considerations

### Dialog Scaling

**Breakpoints**:
- **Large**: >1400px - Full featured interface
- **Medium**: 1000-1400px - Tabbed interface with scrolling
- **Small**: 800-1000px - Wizard-style progression
- **Mini**: <800px - Essential controls only

**Adaptive Elements**:
- Collapsible sections for non-critical options
- Contextual menus instead of permanent buttons
- Scrollable regions for content overflow
- Resizable panels for user customization

### Accessibility Features

**Keyboard Navigation**:
- Tab order follows logical workflow progression
- Arrow keys for grid/list navigation
- Spacebar for toggle actions
- Enter for primary actions, Escape for cancel

**Screen Reader Support**:
- Proper ARIA labels for all interactive elements
- Status announcements for progress updates
- Alternative text for visual progress indicators
- Logical heading structure for dialog sections

**High Contrast Support**:
- Respect GIMP theme settings
- Sufficient color contrast ratios
- Non-color dependent status indicators
- Scalable UI elements for zoom compatibility

## 🎨 Visual Design Language

### Icon System

**Primary Icons** (16x16 and 24x24):
```
🔍 Auto-detect        🌐 Translate        ✍️ Typeset
📖 Extract Text       🖌️ Remove Text      ⚙️ Settings
📋 Review             ↩️ Undo             🔄 Refresh
⏸️ Pause              ✅ Accept           ❌ Cancel
```

**Status Icons**:
```
✓ Success (Green)     ⚠️ Warning (Orange)   ❌ Error (Red)
⏳ Processing (Blue)  📊 Progress (Blue)    💭 Info (Gray)
```

### Color Coding System

**Bubble State Colors**:
- **Detected**: Blue outline (#0078d4)
- **Processing**: Orange outline (#ff8800)
- **Completed**: Green outline (#107c10)
- **Error**: Red outline (#d13438)
- **Selected**: Purple outline (#5c2d91)

**Text Region Highlighting**:
- **Original Text**: Semi-transparent red overlay
- **Extracted Text**: Semi-transparent blue overlay
- **Translation Preview**: Semi-transparent green overlay
- **Confidence Indicators**: Gradient from red (low) to green (high)

### Animation Guidelines

**Performance Considerations**:
- Smooth 60fps animations for state transitions
- Reduced motion respect for accessibility
- Hardware acceleration where available
- Graceful degradation on older systems

**Animation Types**:
- **Fade In/Out**: Content appearance/disappearance (200ms)
- **Slide**: Panel transitions (300ms)
- **Pulse**: Processing indicators (1000ms loop)
- **Bounce**: Success confirmations (400ms)
- **Shake**: Error indications (300ms)

This comprehensive UI design document ensures the manga translator plugin provides a native, intuitive experience within GIMP while supporting users from beginner to expert levels. The design emphasizes workflow efficiency, visual clarity, and robust error handling to create a professional translation tool that integrates seamlessly with GIMP's existing interface paradigms.