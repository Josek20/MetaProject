function load_data(data_path::String)
    train_data = JSON.parsefile(data_path)
end

function preprosses_data_to_expressions(raw_data)
    new_data = []
    for i in raw_data
        push!(new_data, Meta.parse(i[1]))
    end
    return new_data
end
