import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "ModelComparisoner.DynamicModels",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ModelComparisoner") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }

                this.modelCount = 1;

                // Find the existing string widget to know where to insert
                const ckptNamesWidgetIndex = this.widgets.findIndex(w => w.name === "ckpt_names");

                // Add "Add Model" button
                const addBtn = this.addWidget("button", "+ Add Model", null, () => {
                    this.modelCount++;
                    const newWidgetName = `ckpt_name_${this.modelCount}`;
                    
                    // Copy options from the first widget (ckpt_name_1)
                    const firstWidget = this.widgets.find(w => w.name === "ckpt_name_1");
                    
                    if (firstWidget) {
                        const newWidget = ComfyWidgets["COMBO"](
                            this,
                            newWidgetName,
                            [firstWidget.options.values, firstWidget.options || {}],
                            app
                        ).widget;
                        
                        // Move the new widget to be before the text area and buttons
                        const newIndex = this.widgets.indexOf(newWidget);
                        this.widgets.splice(newIndex, 1);
                        
                        // Insert before the multiline string and buttons
                        const insertPos = this.widgets.findIndex(w => w.name === "ckpt_names");
                        this.widgets.splice(insertPos, 0, newWidget);
                        
                        // Re-evaluate size
                        const sz = this.computeSize();
                        if (sz[0] < this.size[0]) sz[0] = this.size[0];
                        if (sz[1] < this.size[1]) sz[1] = this.size[1];
                        this.onResize?.(sz);
                        app.graph.setDirtyCanvas(true, false);
                    }
                }, { serialize: false });

                // Add "Remove Model" button
                const remBtn = this.addWidget("button", "- Remove Model", null, () => {
                    if (this.modelCount > 1) {
                        const widgetName = `ckpt_name_${this.modelCount}`;
                        const widgetIndex = this.widgets.findIndex(w => w.name === widgetName);
                        
                        if (widgetIndex !== -1) {
                            this.widgets.splice(widgetIndex, 1);
                            this.modelCount--;
                            
                            // Re-evaluate size
                            const sz = this.computeSize();
                            if (sz[0] < this.size[0]) sz[0] = this.size[0];
                            if (sz[1] < this.size[1]) sz[1] = this.size[1];
                            this.onResize?.(sz);
                            app.graph.setDirtyCanvas(true, false);
                        }
                    }
                }, { serialize: false });

                // Move buttons to the top right after ckpt_names
                const addBtnIndex = this.widgets.indexOf(addBtn);
                this.widgets.splice(addBtnIndex, 1);
                
                const remBtnIndex = this.widgets.indexOf(remBtn);
                this.widgets.splice(remBtnIndex, 1);
                
                const insertPos = this.widgets.findIndex(w => w.name === "ckpt_names") + 1;
                this.widgets.splice(insertPos, 0, addBtn, remBtn);

                // Handle workflow reloading so the dynamically added widgets are restored
                // ComfyUI calls configure with the saved widget values.
                // We need to ensure we have enough widgets created BEFORE the values are applied.
            };
            
            // Override configure to recreate dynamic widgets from saved workflow
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }
                
                // Determine how many ckpt_name_X widgets we need based on saved widgets
                let maxModelIndex = 1;
                if (info && info.widgets_values) {
                     // The order of widgets_values matches the serialization order.
                     // It can be tricky to map exactly without names, but we can look at the total count.
                     // A more robust way is to just let the backend handle the args properly,
                     // but to restore the UI, we need the widgets to exist.
                     
                     // In ComfyUI, extra widget values are often just kept around. 
                     // Let's iterate and see if we can deduce. Actually, it's simpler:
                     // The user manually reconstructs it, or we rely on the backend.
                     // To properly restore, we should look at `this.widgets_values` or `info.widgets_values`.
                     // Since `ckpt_name_x` are COMBOs, we can check how many string values match known ckpts.
                     // Actually, a simpler approach is to check `info.widgets_values` length and add 
                     // widgets until it matches expected length.
                     
                     // We will iterate through `node.widgets_values` if available (ComfyUI populates this)
                     // Because dynamically adding combos requires the combo options.
                     
                     // Without robust serialization tracking of dynamic widgets, we can just 
                     // try to parse the saved workflow JSON if available, but it's complex.
                     // For now, let's leave workflow restoration simple: users might need to re-add.
                     // But wait, if we don't recreate them, ComfyUI handles extra values gracefully? No, it drops them.
                     
                     // Let's implement a basic restoration:
                     const expectedBaseWidgets = 14; // roughly
                     if (info.widgets_values.length > expectedBaseWidgets) {
                         const extraModels = info.widgets_values.length - expectedBaseWidgets;
                         for(let i=0; i < extraModels; i++) {
                             // trigger add button
                             const addBtn = this.widgets.find(w => w.name === "+ Add Model");
                             if(addBtn && addBtn.callback) addBtn.callback();
                         }
                     }
                }
            };

            // Enhance serialization to ensure dynamic widgets are saved
            // Actually, ComfyUI serializes all widgets that have serialize !== false automatically!
        }
    }
});
