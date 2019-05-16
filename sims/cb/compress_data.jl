press_data(T::Type, y) = [T.(yi) for yi in y]
compress_data(y) = press_data(Float16, y)
decompress_data(y) = press_data(Float64, y)
