using GLMakie
using Graphs
using CairoMakie
using GraphMakie
GLMakie.activate!()


# function plot_radial_interactive_tree(soltree, history)
#     fig, ax, p = plot_radial_interactive_tree(soltree)
#     on(events(p.plots[3]).click) do event
#         node_index = event.index
#         selected_node[] = node_index
#         lineplot[1][] = history[node_index]  # update data
#     end
#     fig
# end

function plot_radial_interactive_tree(soltree, history)
    root_index = findfirst(x->x.depth == 0, collect(values(soltree)))
    root = collect(values(soltree))[root_index]
    g = DiGraph(length(soltree))
    id2gid = Dict(n.node_id => ind for (ind, n) in enumerate(values(soltree)))
    gid2id = Dict(i => ind for (ind, i) in id2gid)

    for node in values(soltree)
        if node.parent != node.node_id
            add_edge!(g, id2gid[node.parent], id2gid[node.node_id])
        end
    end

    # Build children map
    children_map = Dict{Int, Vector{Int}}()
    for node in values(soltree)
        if node.parent != node.node_id
            push!(get!(children_map, id2gid[node.parent], Int[]), id2gid[node.node_id])
        end
    end

    # Recursive function to assign radial layout positions
    function assign_radial_pos(node_id, depth, start_angle, end_angle, positions)
        # Radius based on depth
        radius = depth * 12.0
        # Angle in radians is mid of allocated angular span
        angle = (start_angle + end_angle) / 2
        # Cartesian coordinates from polar coords
        x = radius * cos(angle)
        y = radius * sin(angle)
        positions[node_id] = Point2f(x, y)

        children = get(children_map, node_id, [])
        n = length(children)
        if n > 0
            angle_span = end_angle - start_angle
            # Divide angular span evenly among children
            angle_step = angle_span / n
            for (i, child) in enumerate(children)
                child_start = start_angle + (i-1)*angle_step
                child_end = child_start + angle_step
                assign_radial_pos(child, depth+1, child_start, child_end, positions)
            end
        end
    end

    # Find root id (node whose parent == node_id)
    root_node = root.node_id
    root_gid = id2gid[soltree[root_node].node_id]

    positions = Dict{Int, Point2f}()
    assign_radial_pos(root_gid, 0, 0, 2pi, positions)

    # Build layout vector matching graph vertices
    layout = [positions[i] for i in 1:nv(g)]

    fig = Figure(resolution=(1200,1200))
    ax = Axis(fig[1,1]; aspect=1)
    # ax = Axis(fig[1,1]; aspect=DataAspect(), autolimits=false)
    # ax.xlimits = (-80, 80)
    # ax.ylimits = (-80, 80)
    fig, ax, p = graphplot(g; layout=layout, edge_linestyle=:solid, node_color=:lightgray, edge_color=:lightgray, node_size=12)

    comments = Dict()
    for (i,pos) in enumerate(layout)
        comments[i] =  string(MyModule.expr(MyModule.nc, soltree[gid2id[i]].ex))
    end
    mahistory = Dict()
    for (i,pos) in enumerate(layout)
        comments[i] =  string(MyModule.expr(MyModule.nc, soltree[gid2id[i]].ex))
    end
    # @assert 0x7356776a85238df8 in keys(history)
    p.plots[3].inspector_label = (plot, index, position) -> begin
        text = comments[index]
        # text = replace_urls(text)
        # text = wrap_text(text, 80)
        # remove_emojis(join(text,"\n\n"))
        # join(text,"\n\n")
        @show gid2id[index]
        final_string = text * "| $(last(history[gid2id[index]]))"
        @show final_string
        final_string
    end
    p.plots[1].inspector_label = (plot, index, position) -> begin
        ""
    end
    inspector = DataInspector(fig);
    return fig, ax, p
end