
// Global variables
let cur_scene = null;
let mouseTrajectory = [];
let featureTrajectory = [];

// Calculate Euclidean distance between two points
function distance_sq(left, right) {
    const dx = left.x - right.x;
    const dy = left.y - right.y;
    return dx * dx + dy * dy;
}

function distance(left, right) {
    return Math.sqrt(distance_sq(left, right));
}


// Represents a circular node in the animal structure
class Circle {
    constructor(index, center, radius, color,
        is_fixed = false) {
        this.index = index;
        this.x = center.x;
        this.y = center.y
        this.radius = radius;
        this.radius_sq = radius * radius;
        this.color = color;
        this.is_fixed = is_fixed;
        this.dx = 0;
        this.dy = 0;
        this.validator = null;
    }

    // Method to plot a circle with a specific color
    plotWithColor(ctx, color) {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
    }

    // Check if a point is inside the circle
    is_in(other_pt) {
        return distance_sq(this, other_pt) <= this.radius_sq;
    }

    // Set movement validator for this node
    set_validator(validator) {
        this.validator = validator;
    }

    // Move the circle by a delta amount, applying validation if set
    move(delta) {
        if (this.validator != null) {
            delta = this.validator.validate(this, delta);
        }
        this.x += delta.x;
        this.y += delta.y;
        return (delta)
    }

    // Draw the circle on the canvas context
    plot(ctx) {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, 2 * Math.PI);
        ctx.fillStyle = this.color;
        ctx.fill();
    }
}

// Validates and constrains node movement within specified ranges
class Validator {
    constructor(base_coord, angle, length_range) {
        this.base_coord = base_coord;
        this.angle = angle;
        this.length_min = length_range.min;
        this.length_max = length_range.max;
    }

    // Update the reference point coordinates
    set_base(base_coord) {
        this.base_coord = base_coord;
    }

    // Validate and adjust movement to stay within constraints
    validate(cur_coord, delta) {
        const next_coord = {
            x: cur_coord.x + delta.x,
            y: cur_coord.y + delta.y
        };
        
        const length = distance(this.base_coord, next_coord);

        // Constrain length and angle to allowed ranges
        const constrained_length = Math.max(this.length_min, Math.min(this.length_max, length));

        // Calculate adjusted movement
        return {
            x: this.base_coord.x - cur_coord.x + 
               constrained_length * Math.cos(this.angle),
            y: this.base_coord.y - cur_coord.y + 
               constrained_length * Math.sin(this.angle)
        };
    }
}

// Represents an interactive animal figure that can be manipulated
class Animal {
    constructor(canvas, type, info, iTrial, body_length = 36, node_radius = 4, line_width = 9) {
        // Basic configuration
        this.canvas = $("#" + canvas)[0]; // Must have jQuery package imported!
        this.type = type; // 1 = stable; 2 = dynamic
        this.info = info; // information of initial animal
        this.iTrial = iTrial; 

        // Canvas setup
        this.ctx = this.canvas.getContext("2d");
        this.width = this.canvas.width;
        this.height = this.canvas.height;
        this.BB = this.canvas.getBoundingClientRect();
        this.offsetX = this.BB.left;
        this.offsetY = this.BB.top;

        // Interaction states
        this.dragok = false;
        this.startX = 0;
        this.startY = 0;

        // Visual properties
        this.node_radius = node_radius;
        this.line_width = line_width;
        this.body_length = body_length;

        this.angle = Math.PI / 4; 


        // === Photo config ===
        this.photoWidth = 36; // 短边像素
        this.centerSize = 61; // 中心图片大小

        this.photos = {
            f1: new Image(),
            f2: new Image(),
            f3: new Image(),
            f4: new Image(),
        };

        // 加载完成就重画一次（防止第一次 draw 时图片还没ready）
        Object.entries({
            f1: "feature-yellow.png",
            f2: "feature-pink.png",
            f3: "feature-green.png",
            f4: "feature-blue.png",
        }).forEach(([k, filename]) => {
            this.photos[k].onload = () => this.draw();
            this.photos[k].src = filename;
        });

        this.centerImg = new Image();
        this.centerImg.src = 'center.png'; // Assuming 'center.png' is in the same directory
        this.centerImg.onload = () => this.draw(); // Redraw once the image is loaded


        // Feature range
        this.features_range = {
            f1_length: {
                min: this.body_length,
                max: this.body_length * 4
            },
            f2_length: {
                min: this.body_length,
                max: this.body_length * 4
            },
            f3_length: {
                min: this.body_length,
                max: this.body_length * 4
            },
            f4_length: {
                min: this.body_length,
                max: this.body_length * 4
            }
        };


        // Initial feature
        this.features_init = {
            f1_length: (this.features_range.f1_length.min + this.features_range.f1_length.max)/2,
            f2_length: (this.features_range.f2_length.min + this.features_range.f2_length.max)/2,
            f3_length: (this.features_range.f3_length.min + this.features_range.f3_length.max)/2,
            f4_length: (this.features_range.f4_length.min + this.features_range.f4_length.max)/2,
        }


        this.coor_init = this.transform_feature_to_coor(this.angle, this.features_init);

        this.nodes = {
            f1_start: new Circle(1, {
                x: this.coor_init.x11,
                y: this.coor_init.y11
            }, this.node_radius, "#000", true), // left-up base
            f2_start: new Circle(2, {
                x: this.coor_init.x21,
                y: this.coor_init.y21
            }, this.node_radius, "#000", true), // left-down base
            f3_start: new Circle(3, {
                x: this.coor_init.x31,
                y: this.coor_init.y31
            }, this.node_radius, "#000", true), // right-up base
            f4_start: new Circle(4, {
                x: this.coor_init.x41,
                y: this.coor_init.y41
            }, this.node_radius, "#000", true), // right-down base
            f1_end: new Circle(5, {
                x: this.coor_init.x12,
                y: this.coor_init.y12
            }, this.node_radius, "#000"),
            f2_end: new Circle(6, {
                x: this.coor_init.x22,
                y: this.coor_init.y22
            }, this.node_radius, "#000"),
            f3_end: new Circle(7, {
                x: this.coor_init.x32,
                y: this.coor_init.y32
            }, this.node_radius, "#000"),
            f4_end: new Circle(8, {
                x: this.coor_init.x42,
                y: this.coor_init.y42
            }, this.node_radius, "#000"),
        };

        this.nodes.f1_end.set_validator(new Validator(this.nodes.f1_start,
            Math.PI + this.angle, this.features_range.f1_length));

        this.nodes.f2_end.set_validator(new Validator(this.nodes.f2_start,
            Math.PI - this.angle, this.features_range.f2_length));

        this.nodes.f3_end.set_validator(new Validator(this.nodes.f3_start,
            - this.angle, this.features_range.f3_length));

        this.nodes.f4_end.set_validator(new Validator(this.nodes.f4_start,
            this.angle, this.features_range.f4_length));

        this.canvas.onmousedown = ((e) => this.myDown(e));
        document.onmouseup = ((e) => this.myUp(e));
        this.canvas.onmousemove = ((e) => this.myMove(e));

    }

    // Transform features to coordinates
    transform_feature_to_coor(angle, features) {
        var x11 = this.width / 2 - Math.cos(angle) * this.body_length / 2;
        var y11 = this.height / 2 - Math.sin(angle) * this.body_length / 2;

        var x12 = x11 - Math.cos(angle) * features.f1_length;
        var y12 = y11 - Math.sin(angle) * features.f1_length;

        var x21 = this.width / 2 - Math.cos(angle) * this.body_length / 2;
        var y21 = this.height / 2 + Math.sin(angle) * this.body_length / 2;

        var x22 = x21 - Math.cos(angle) * features.f2_length;
        var y22 = y21 + Math.sin(angle) * features.f2_length;

        var x31 = this.width / 2 + Math.cos(angle) * this.body_length / 2;
        var y31 = this.height / 2 - Math.sin(angle) * this.body_length / 2;

        var x32 = x31 + Math.cos(angle) * features.f3_length;
        var y32 = y31 - Math.sin(angle) * features.f3_length;

        var x41 = this.width / 2 + Math.cos(angle) * this.body_length / 2;
        var y41 = this.height / 2 + Math.sin(angle) * this.body_length / 2;

        var x42 = x41 + Math.cos(angle) * features.f4_length;
        var y42 = y41 + Math.sin(angle) * features.f4_length;

        var coordinate = {
            x11: x11,
            x12: x12,
            x21: x21,
            x22: x22,
            x31: x31,
            x32: x32,
            x41: x41,
            x42: x42,
            y11: y11,
            y12: y12,
            y21: y21,
            y22: y22,
            y31: y31,
            y32: y32,
            y41: y41,
            y42: y42
        };

        return coordinate;
    };

    // Transform coordinates to features
    normalize(value, min, max) {
        // Check if min and max are the same
        if (min === max) {
            return 0.5; // Default value when min and max are the same
        }
        let normalized = (value - min) / (max - min);
        return parseFloat(normalized.toFixed(14));
    }

    transform_coor_to_feature(nodes) {
        // Get raw values first
        const rawFeatures = {
            f1_length: distance(nodes.f1_start, nodes.f1_end),
            f2_length: distance(nodes.f2_start, nodes.f2_end),
            f3_length: distance(nodes.f3_start, nodes.f3_end),
            f4_length: distance(nodes.f4_start, nodes.f4_end)
        };

        // Normalize each feature using the helper function
        return {
            f1_length: this.normalize(rawFeatures.f1_length, 
                this.features_range.f1_length.min, 
                this.features_range.f1_length.max),
            f2_length: this.normalize(rawFeatures.f2_length, 
                this.features_range.f2_length.min, 
                this.features_range.f2_length.max),
            f3_length: this.normalize(rawFeatures.f3_length, 
                this.features_range.f3_length.min, 
                this.features_range.f3_length.max),
            f4_length: this.normalize(rawFeatures.f4_length, 
                this.features_range.f4_length.min, 
                this.features_range.f4_length.max)
        };
    }


    drawPhotoOnSegment(startNode, endNode, img, fixedWidthPx) {
        if (!img || !img.complete) return; // Ensure image is loaded

        const ctx = this.ctx;

        const dx = endNode.x - startNode.x;
        const dy = endNode.y - startNode.y;
        const len = Math.sqrt(dx * dx + dy * dy);

        if (!Number.isFinite(len) || len <= 1) return;

        // 线段方向角
        const angle = Math.atan2(dy, dx);

        // ✅ 关键：让“图片的长边（height方向）”沿着线段方向增长
        // 由于 drawImage 的 dH 是沿“当前坐标系的 +y”方向，
        // 我们把坐标系旋转到：+y 正好指向线段方向
        ctx.save();
        ctx.translate(startNode.x, startNode.y);
        ctx.rotate(angle - Math.PI / 2);

        // 把坐标移到线段末端，并旋转 180°
        ctx.translate(0, len);
        ctx.rotate(Math.PI);

        // 画一个宽度固定、长度随 len 变化的矩形贴图
        // 宽度：fixedWidthPx（不变）
        // 长度：len（随拖动变化）
        ctx.drawImage(img, -fixedWidthPx / 2, 0, fixedWidthPx, len);

        ctx.restore();
    }



    // draw animal
    draw() {
        this.ctx.clearRect(0, 0, this.width, this.height);

        this.drawPhotoOnSegment(this.nodes.f1_start, this.nodes.f1_end, this.photos.f1, this.photoWidth);
        this.drawPhotoOnSegment(this.nodes.f2_start, this.nodes.f2_end, this.photos.f2, this.photoWidth);
        this.drawPhotoOnSegment(this.nodes.f3_start, this.nodes.f3_end, this.photos.f3, this.photoWidth);
        this.drawPhotoOnSegment(this.nodes.f4_start, this.nodes.f4_end, this.photos.f4, this.photoWidth);

        // Draw the center image in the middle of the canvas
        const centerX = this.width / 2 - this.centerSize / 2; 
        const centerY = this.height / 2 - this.centerSize / 2;
        this.ctx.drawImage(this.centerImg, centerX, centerY, this.centerSize, this.centerSize);


        // const ctx = this.ctx;
        // ctx.lineWidth = this.line_width;

        // ctx.beginPath();
        // ctx.moveTo(this.nodes.f1_start.x, this.nodes.f1_start.y);
        // ctx.lineTo(this.nodes.f1_end.x, this.nodes.f1_end.y);
        // ctx.stroke();
        // ctx.beginPath();
        // ctx.moveTo(this.nodes.f2_start.x, this.nodes.f2_start.y);
        // ctx.lineTo(this.nodes.f2_end.x, this.nodes.f2_end.y);
        // ctx.stroke();
        // ctx.beginPath();
        // ctx.moveTo(this.nodes.f3_start.x, this.nodes.f3_start.y);
        // ctx.lineTo(this.nodes.f3_end.x, this.nodes.f3_end.y);
        // ctx.stroke();
        // ctx.beginPath();
        // ctx.moveTo(this.nodes.f4_start.x, this.nodes.f4_start.y);
        // ctx.lineTo(this.nodes.f4_end.x, this.nodes.f4_end.y);
        // ctx.stroke();

        // Draw only the end points of each feature with specific colors
        this.nodes.f1_end.plotWithColor(this.ctx, "#FF8C00"); // Deep yellow (feature 1)
        this.nodes.f2_end.plotWithColor(this.ctx, "#FF1493"); // Deep pink (feature 2)
        this.nodes.f3_end.plotWithColor(this.ctx, "#006400"); // Deep green (feature 3)
        this.nodes.f4_end.plotWithColor(this.ctx, "#00008B"); // Deep blue (feature 4)


        return 0;
    }

    // handle mousedown events
    myDown(e) {
        console.log("Down");
        this.BB = this.canvas.getBoundingClientRect();
        this.offsetX = this.BB.left;
        this.offsetY = this.BB.top;
        // tell the browser we're handling this mouse event
        e.preventDefault();
        e.stopPropagation();

        // get the current mouse position
        var mx = e.clientX - this.offsetX;
        var my = e.clientY - this.offsetY;

        // test each shape to see if mouse is inside
        this.dragok = false;
        if (this.type == 2) {
            for (const i in this.nodes) {
                var block = this.nodes[i];
                if (block.is_in({
                    x: mx,
                    y: my
                })) {
                    this.dragok = true;
                    cur_scene = this;
                    block.isDragging = true;
                }
            }            
        }

        // save the current mouse position
        this.startX = mx;
        this.startY = my;

        mouseTrajectory.push({x:parseFloat(mx.toFixed(14)), y:parseFloat(my.toFixed(14)), action:"down", timestamp:Date.now()});

    }

    // handle mouseup events
    myUp(e) {
        if (!this.dragok)
            return;
        console.log("Up");
        //if (!this.dragok) return;
        // tell the browser we're handling this mouse event
        e.preventDefault();
        e.stopPropagation();

        // get the current mouse position
        var mx = e.clientX - this.offsetX;
        var my = e.clientY - this.offsetY;

        // clear all the dragging flags
        this.dragok = false;
        cur_scene = null;
        for (const i in this.nodes) {
            this.nodes[i].isDragging = false;
        }

        mouseTrajectory.push({x:parseFloat(mx.toFixed(14)), y:parseFloat(my.toFixed(14)), action:"up", timestamp:Date.now()});

    }

    // handle mouse moves
    myMove(e) {
        // if we're dragging anything...
        if (this.dragok) {
            console.log("Dragging");
            // tell the browser we're handling this mouse event
            e.preventDefault();
            e.stopPropagation();

            // get the current mouse position
            var mx = e.clientX - this.offsetX;
            var my = e.clientY - this.offsetY;

            // calculate the distance the mouse has moved
            // since the last mousemove
            var dx = mx - this.startX;
            var dy = my - this.startY;

            // move each node that isDragging
            // by the distance the mouse has moved
            // since the last mousemove
            var index_is_dragging = -1;
            for (const i in this.nodes) {
                var target = this.nodes[i];
                if (target.is_fixed) {
                    continue;
                }
                if (target.isDragging) {
                    index_is_dragging = i;
                    var delta = target.move({ x: dx, y: dy }); // the real restricted distance
                    var nodex = target.x;
                    var nodey = target.y;
                    // console.log(index_is_dragging, nodex, nodey);
                }
            }

            // After drawing, record feature values
            if (index_is_dragging !== -1) {
                const features = this.transform_coor_to_feature(this.nodes);
                const featureRecord = 
                    {
                        f1_length: features.f1_length,
                        f2_length: features.f2_length,
                        f3_length: features.f3_length,
                        f4_length: features.f4_length,
                        timestamp: Date.now()
                    };
                featureTrajectory.push(featureRecord);
            }            

            // redraw the scene with the new node positions
            this.draw();
            // reset the starting mouse position for the next mousemove
            this.startX = mx;
            this.startY = my;
                        
            mouseTrajectory.push({x:parseFloat(nodex.toFixed(14)), y:parseFloat(nodey.toFixed(14)), action:index_is_dragging, timestamp:Date.now()});
    
        }
    }

}
