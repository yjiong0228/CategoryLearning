
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
        children = [], is_fixed = false) {
        this.index = index;
        this.x = center.x;
        this.y = center.y
        this.radius = radius;
        this.radius_sq = radius * radius;
        this.color = color;
        this.children = children;
        this.is_fixed = is_fixed;
        this.dx = 0;
        this.dy = 0;
        this.validator = null;
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
    constructor(base_coord, angle_range, length_range, orientation) {
        this.base_coord = base_coord;
        this.angle_min = angle_range.min;
        this.angle_max = angle_range.max;
        this.length_min = length_range.min;
        this.length_max = length_range.max;
        this.orientation = orientation;
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
        const angle = Math.asin((next_coord.y - this.base_coord.y) / length);

        // Constrain length and angle to allowed ranges
        const constrained_length = Math.max(this.length_min, 
                                          Math.min(this.length_max, length));
        const constrained_angle = Math.max(this.angle_min, 
                                         Math.min(this.angle_max, angle));

        // Calculate adjusted movement
        return {
            x: this.base_coord.x - cur_coord.x + 
               constrained_length * Math.cos(constrained_angle) * this.orientation,
            y: this.base_coord.y - cur_coord.y + 
               constrained_length * Math.sin(constrained_angle)
        };
    }
}

// Represents an interactive animal figure that can be manipulated
class Animal {
    constructor(version, task, canvas, type, info, iTrial, body_length = 130, node_radius = 5, line_width = 9) {
        // Basic configuration
        this.version = version; // 1 = learn length; 2 = learn angle
        this.task = task; // 1 = Task1a; 2 = Task1b & Task3d
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
        this.flipped = null;

        // Initial orientation
        if (this.flipped != null) {
            this.orientation = this.flip();
        } else {
            if (this.task == 1) {
                this.orientation = Math.round(Math.random()) * 2 - 1;
            } else if (this.task == 2) {
                this.orientation = this.info.body_ori[this.iTrial-1];
            }
        }

        // Feature range
        if (this.version == '1') {
            this.features_range = {
                neck_length: {
                    min: this.body_length / 4,
                    max: this.body_length * 5 / 4
                },
                neck_angle: {
                    min: -Math.PI / 6,
                    max: -Math.PI / 6
                },
                head_length: {
                    min: this.body_length / 4,
                    max: this.body_length * 5 / 4
                },
                head_angle: {
                    min: Math.PI / 6.,
                    max: Math.PI / 6.,
                },
                leg_length: {
                    min: this.body_length / 4,
                    max: this.body_length * 5 / 4
                },
                leg_angle: {
                    min: Math.PI * 14 / 36.,
                    max: Math.PI * 14 / 36.
                },
                tail_length: {
                    min: this.body_length / 4,
                    max: this.body_length * 5 / 4
                },
                tail_angle: {
                    min: -Math.PI / 6.,
                    max: -Math.PI / 6.
                },
            };
        } else if (this.version == '2') {
            this.features_range = {
                neck_length: {
                    min: this.body_length * 3 / 4,
                    max: this.body_length * 3 / 4
                },
                neck_angle: {
                    min: -Math.PI / 3,
                    max: 0.
                },
                head_length: {
                    min: this.body_length * 3 / 4,
                    max: this.body_length * 3 / 4
                },
                head_angle: {
                    min: 0.,
                    max: Math.PI / 3
                },
                leg_length: {
                    min: this.body_length * 3 / 4,
                    max: this.body_length * 3 / 4
                },
                leg_angle: {
                    min: Math.PI * 11 / 36.,
                    max: Math.PI * 17 / 36.
                },
                tail_length: {
                    min: this.body_length * 3 / 4,
                    max: this.body_length * 3 / 4
                },
                tail_angle: {
                    min: -Math.PI / 3.,
                    max: 0.
                },
            };
        }

        // Initial feature
        if (this.task == 1) {
            this.features_init = {
                neck_length: (this.features_range.neck_length.min + this.features_range.neck_length.max)/2,
                neck_angle: (this.features_range.neck_angle.min + this.features_range.neck_angle.max)/2,
                head_length: (this.features_range.head_length.min + this.features_range.head_length.max)/2,
                head_angle: (this.features_range.head_angle.min + this.features_range.head_angle.max)/2,
                leg_length: (this.features_range.leg_length.min + this.features_range.leg_length.max)/2,
                leg_angle: (this.features_range.leg_angle.min + this.features_range.leg_angle.max)/2,
                tail_length: (this.features_range.tail_length.min + this.features_range.tail_length.max)/2,
                tail_angle: (this.features_range.tail_angle.min + this.features_range.tail_angle.max)/2.
            }
        } else if (this.task == 2) {
            this.features_init = {
                neck_length: this.features_range.neck_length.min + this.info.neck_length[this.iTrial-1] * (this.features_range.neck_length.max - this.features_range.neck_length.min),
                neck_angle: this.features_range.neck_angle.min + this.info.neck_angle[this.iTrial-1] * (this.features_range.neck_angle.max - this.features_range.neck_angle.min),
                head_length: this.features_range.head_length.min + this.info.head_length[this.iTrial-1] * (this.features_range.head_length.max - this.features_range.head_length.min),
                head_angle: this.features_range.head_angle.min + this.info.head_angle[this.iTrial-1] * (this.features_range.head_angle.max - this.features_range.head_angle.min),
                leg_length: this.features_range.leg_length.min + this.info.leg_length[this.iTrial-1] * (this.features_range.leg_length.max - this.features_range.leg_length.min),
                leg_angle: this.features_range.leg_angle.min + this.info.leg_angle[this.iTrial-1] * (this.features_range.leg_angle.max - this.features_range.leg_angle.min),
                tail_length: this.features_range.tail_length.min + this.info.tail_length[this.iTrial-1] * (this.features_range.tail_length.max - this.features_range.tail_length.min),
                tail_angle: this.features_range.tail_angle.min + this.info.tail_angle[this.iTrial-1] * (this.features_range.tail_angle.max - this.features_range.tail_angle.min),
            }
        }

        this.coor_init = this.transform_feature_to_coor(this.orientation, this.features_init);

        this.nodes = {
            head_end: new Circle(0, {
                x: this.coor_init.x0,
                y: this.coor_init.y0
            }, this.node_radius, "#000"),
            head_neck: new Circle(1, {
                x: this.coor_init.x1,
                y: this.coor_init.y1
            }, this.node_radius, "#000", ["head_end"]),
            neck_body_leg: new Circle(2, {
                x: this.coor_init.x2,
                y: this.coor_init.y2
            }, this.node_radius, "#000", [], true),
            body_leg_tail: new Circle(3, {
                x: this.coor_init.x3,
                y: this.coor_init.y3
            }, this.node_radius, "#000", [], true),
            leg_end_0: new Circle(4, {
                x: this.coor_init.x4,
                y: this.coor_init.y4
            }, this.node_radius, "#000"),
            leg_end_1: new Circle(5, {
                x: this.coor_init.x5,
                y: this.coor_init.y5
            }, this.node_radius, "#000"),
            leg_end_2: new Circle(6, {
                x: this.coor_init.x6,
                y: this.coor_init.y6
            }, this.node_radius, "#000"),
            leg_end_3: new Circle(7, {
                x: this.coor_init.x7,
                y: this.coor_init.y7
            }, this.node_radius, "#000"),
            tail_end: new Circle(8, {
                x: this.coor_init.x8,
                y: this.coor_init.y8
            }, this.node_radius, "#000"),
        };

        this.nodes.head_end.set_validator(new Validator(this.nodes.head_neck,
            this.features_range.head_angle, this.features_range.head_length,
            this.orientation));

        this.nodes.head_neck.set_validator(new Validator(this.nodes.neck_body_leg,
            this.features_range.neck_angle, this.features_range.neck_length,
            this.orientation));

        this.nodes.leg_end_0.set_validator(new Validator(this.nodes.neck_body_leg,
            this.features_range.leg_angle, this.features_range.leg_length,
            this.orientation));

        this.nodes.leg_end_1.set_validator(new Validator(this.nodes.neck_body_leg,
            this.features_range.leg_angle, this.features_range.leg_length,
            -this.orientation));

        this.nodes.leg_end_2.set_validator(new Validator(this.nodes.body_leg_tail,
            this.features_range.leg_angle, this.features_range.leg_length,
            this.orientation));

        this.nodes.leg_end_3.set_validator(new Validator(this.nodes.body_leg_tail,
            this.features_range.leg_angle, this.features_range.leg_length,
            -this.orientation));

        this.nodes.tail_end.set_validator(new Validator(this.nodes.body_leg_tail,
            this.features_range.tail_angle, this.features_range.tail_length,
            -this.orientation));

        this.canvas.onmousedown = ((e) => this.myDown(e));
        document.onmouseup = ((e) => this.myUp(e));
        this.canvas.onmousemove = ((e) => this.myMove(e));

    }

    // Transform features to coordinates
    transform_feature_to_coor(orientation, features) {
        var x2 = this.width / 2 + orientation * this.body_length / 2;
        var y2 = this.height / 2;

        var x3 = this.width / 2 - orientation * this.body_length / 2;
        var y3 = y2;

        var x1 = x2 + orientation * features.neck_length * Math.cos(features.neck_angle);
        var y1 = y2 + features.neck_length * Math.sin(features.neck_angle);

        var x0 = x1 + orientation * features.head_length * Math.cos(features.head_angle);
        var y0 = y1 + features.head_length * Math.sin(features.head_angle);

        var x8 = x3 - orientation * features.tail_length * Math.cos(features.tail_angle);
        var y8 = y3 + features.tail_length * Math.sin(features.tail_angle);

        var x4 = x2 + orientation * features.leg_length * Math.cos(features.leg_angle);
        var y4 = y2 + features.leg_length * Math.sin(features.leg_angle);
        var x5 = x2 - orientation * features.leg_length * Math.cos(features.leg_angle);
        var y5 = y2 + features.leg_length * Math.sin(features.leg_angle);

        var x6 = x3 + orientation * features.leg_length * Math.cos(features.leg_angle);
        var y6 = y3 + features.leg_length * Math.sin(features.leg_angle);
        var x7 = x3 - orientation * features.leg_length * Math.cos(features.leg_angle);
        var y7 = y3 + features.leg_length * Math.sin(features.leg_angle);

        var coordinate = {
            x0: x0,
            x1: x1,
            x2: x2,
            x3: x3,
            x4: x4,
            x5: x5,
            x6: x6,
            x7: x7,
            x8: x8,
            y0: y0,
            y1: y1,
            y2: y2,
            y3: y3,
            y4: y4,
            y5: y5,
            y6: y6,
            y7: y7,
            y8: y8,
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
        const getAngle = (p1, p2) => Math.asin((p1.y - p2.y) / distance(p1, p2));
        
        // Get raw values first
        const rawFeatures = {
            neck_length: distance(nodes.neck_body_leg, nodes.head_neck),
            neck_angle: getAngle(nodes.head_neck, nodes.neck_body_leg),
            head_length: distance(nodes.head_neck, nodes.head_end),
            head_angle: getAngle(nodes.head_end, nodes.head_neck),
            leg_length: distance(nodes.leg_end_0, nodes.neck_body_leg),
            leg_angle: getAngle(nodes.leg_end_0, nodes.neck_body_leg),
            tail_length: distance(nodes.body_leg_tail, nodes.tail_end),
            tail_angle: getAngle(nodes.tail_end, nodes.body_leg_tail)
        };

        // Normalize each feature using the helper function
        return {
            neck_length: this.normalize(rawFeatures.neck_length, 
                this.features_range.neck_length.min, 
                this.features_range.neck_length.max),
            neck_angle: this.normalize(rawFeatures.neck_angle, 
                this.features_range.neck_angle.min, 
                this.features_range.neck_angle.max),
            head_length: this.normalize(rawFeatures.head_length, 
                this.features_range.head_length.min, 
                this.features_range.head_length.max),
            head_angle: this.normalize(rawFeatures.head_angle, 
                this.features_range.head_angle.min, 
                this.features_range.head_angle.max),
            leg_length: this.normalize(rawFeatures.leg_length, 
                this.features_range.leg_length.min, 
                this.features_range.leg_length.max),
            leg_angle: this.normalize(rawFeatures.leg_angle, 
                this.features_range.leg_angle.min, 
                this.features_range.leg_angle.max),
            tail_length: this.normalize(rawFeatures.tail_length, 
                this.features_range.tail_length.min, 
                this.features_range.tail_length.max),
            tail_angle: this.normalize(rawFeatures.tail_angle, 
                this.features_range.tail_angle.min, 
                this.features_range.tail_angle.max)
        };
    }

    // Synchronize leg changes
    adjust_node_locations(reference = 4) {
        var offset = {
            x: 0,
            y: 0
        };
        switch (reference) {
            case 4:
                offset = {
                    x: -(this.nodes.neck_body_leg.x -
                        this.nodes.leg_end_0.x) * this.orientation,
                    y: -(this.nodes.neck_body_leg.y -
                        this.nodes.leg_end_0.y),
                };
                break;
            case 5:
                offset = {
                    x: (this.nodes.neck_body_leg.x -
                        this.nodes.leg_end_1.x) * this.orientation,
                    y: -(this.nodes.neck_body_leg.y -
                        this.nodes.leg_end_1.y),
                };
                break;
            case 6:
                offset = {
                    x: -(this.nodes.body_leg_tail.x -
                        this.nodes.leg_end_2.x) * this.orientation,
                    y: -(this.nodes.body_leg_tail.y -
                        this.nodes.leg_end_2.y),
                };
                break;
            case 7:
                offset = {
                    x: (this.nodes.body_leg_tail.x -
                        this.nodes.leg_end_3.x) * this.orientation,
                    y: -(this.nodes.body_leg_tail.y -
                        this.nodes.leg_end_3.y),
                };
        }
        this.nodes.leg_end_0.x = this.nodes.neck_body_leg.x +
            this.orientation * offset.x;
        this.nodes.leg_end_1.x = this.nodes.neck_body_leg.x -
            this.orientation * offset.x;
        this.nodes.leg_end_2.x = this.nodes.body_leg_tail.x +
            this.orientation * offset.x;
        this.nodes.leg_end_3.x = this.nodes.body_leg_tail.x -
            this.orientation * offset.x;
        this.nodes.leg_end_0.y = this.nodes.neck_body_leg.y +
            offset.y;
        this.nodes.leg_end_1.y = this.nodes.neck_body_leg.y +
            offset.y;
        this.nodes.leg_end_2.y = this.nodes.body_leg_tail.y +
            offset.y;
        this.nodes.leg_end_3.y = this.nodes.body_leg_tail.y +
            offset.y;

    }

    // flip the animal
    flip() {
        console.log("Flip");

        this.flipped = 1;
        this.orientation = - this.orientation;

        for (var node_id in this.nodes) {
            if (this.nodes[node_id].validator != null)
                this.nodes[node_id].validator.orientation *= -1;
            this.nodes[node_id].x = this.width - this.nodes[node_id].x;
        }

        this.draw();
        mouseTrajectory.push({x:"NA", y:"NA", action:"flip", timestamp:Date.now()});
        return this.orientation;
    }

    // draw animal
    draw() {
        this.ctx.clearRect(0, 0, this.width, this.height);

        for (const node in this.nodes) {
            this.nodes[node].plot(this.ctx);
        }
        var ctx = this.ctx;
        ctx.lineWidth = this.line_width;

        ctx.beginPath();
        ctx.moveTo(this.nodes.head_end.x, this.nodes.head_end.y);
        ctx.lineTo(this.nodes.head_neck.x, this.nodes.head_neck.y);
        ctx.lineTo(this.nodes.neck_body_leg.x, this.nodes.neck_body_leg.y);
        ctx.lineTo(this.nodes.body_leg_tail.x, this.nodes.body_leg_tail.y);
        ctx.lineTo(this.nodes.tail_end.x, this.nodes.tail_end.y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(this.nodes.leg_end_0.x, this.nodes.leg_end_0.y);
        ctx.lineTo(this.nodes.neck_body_leg.x, this.nodes.neck_body_leg.y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(this.nodes.neck_body_leg.x, this.nodes.neck_body_leg.y);
        ctx.lineTo(this.nodes.leg_end_1.x, this.nodes.leg_end_1.y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(this.nodes.leg_end_2.x, this.nodes.leg_end_2.y);
        ctx.lineTo(this.nodes.body_leg_tail.x, this.nodes.body_leg_tail.y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(this.nodes.body_leg_tail.x, this.nodes.body_leg_tail.y);
        ctx.lineTo(this.nodes.leg_end_3.x, this.nodes.leg_end_3.y);
        ctx.stroke();
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
        if (this.task == 1) {
            mouseTrajectory.push({x:parseFloat(mx.toFixed(14)), y:parseFloat(my.toFixed(14)), action:"down", timestamp:Date.now()});
        } else if (this.task == 2) {
            mouseTrajectory.push({iTrial:this.iTrial, x:parseFloat(mx.toFixed(14)), y:parseFloat(my.toFixed(14)), action:"down", timestamp:Date.now()});
        }
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

        if (this.task == 1) {
            mouseTrajectory.push({x:parseFloat(mx.toFixed(14)), y:parseFloat(my.toFixed(14)), action:"up", timestamp:Date.now()});
        } else if (this.task == 2) {
            mouseTrajectory.push({iTrial:this.iTrial, x:parseFloat(mx.toFixed(14)), y:parseFloat(my.toFixed(14)), action:"up", timestamp:Date.now()});
        }
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
                    if (target.index >= 4 && target.index <= 7) {
                        this.adjust_node_locations(target.index);
                    }
                    for (var j = 0; j < target.children.length; j++) {
                        this.nodes[target.children[j]].move(delta);
                    }
                }
            }

            // After drawing, record feature values
            if (index_is_dragging !== -1) {
                const features = this.transform_coor_to_feature(this.nodes);
                const featureRecord = this.task === 1 ? 
                    {
                        neck_length: features.neck_length,
                        head_length: features.head_length,
                        leg_length: features.leg_length,
                        tail_length: features.tail_length,
                        neck_angle: features.neck_angle,
                        head_angle: features.head_angle,
                        leg_angle: features.leg_angle,
                        tail_angle: features.tail_angle,
                        timestamp: Date.now()
                    } :
                    {
                        iTrial: this.iTrial,
                        neck_length: features.neck_length,
                        head_length: features.head_length,
                        leg_length: features.leg_length,
                        tail_length: features.tail_length,
                        neck_angle: features.neck_angle,
                        head_angle: features.head_angle,
                        leg_angle: features.leg_angle,
                        tail_angle: features.tail_angle,
                        timestamp: Date.now()
                    };
                featureTrajectory.push(featureRecord);
            }            

            // redraw the scene with the new node positions
            this.draw();
            // reset the starting mouse position for the next mousemove
            this.startX = mx;
            this.startY = my;
                        
            if (this.task == 1) {
                mouseTrajectory.push({x:parseFloat(nodex.toFixed(14)), y:parseFloat(nodey.toFixed(14)), action:index_is_dragging, timestamp:Date.now()});
            } else if (this.task == 2) {
                mouseTrajectory.push({iTrial:this.iTrial, x:parseFloat(nodex.toFixed(14)), y:parseFloat(nodey.toFixed(14)), action:index_is_dragging, timestamp:Date.now()});
            }
    
        }
    }

}
