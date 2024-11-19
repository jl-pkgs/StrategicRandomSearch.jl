# 定义目标函数
function Zakharov(x::AbstractMatrix{Float64})
  n_rows, n_cols = size(x)
  y1 = zeros(Float64, n_rows)  # 初始化为零向量
  y2 = zeros(Float64, n_rows)  # 初始化为零向量

  @inbounds for i in 1:n_cols
    xi = @view x[:, i]  # 使用视图来引用 x 的第 i 列
    xi_shifted = xi .- 0.5  # 减去 0.5 的操作
    y1 .+= xi_shifted .^ 2  # 累加平方项
    y2 .+= xi_shifted .* (i + 1)  # 累加加权项
  end

  y = y1 .+ (0.5 .* y2) .^ 2 .+ (0.5 .* y2) .^ 4 .+ 1
  return y
end


function Zakharov(x::AbstractVector{Float64})
  x .-= 0.5
  n = length(x)

  # 计算 y1 和 y2
  y1 = sum(x .^ 2)  # 向量化计算 x[i]^2
  y2 = 0.0
  for i in 1:n
    y2 += (i + 1) * x[i]  # 直接计算 y2
  end

  # 计算目标函数值
  y2_half = 0.5 * y2
  y = y1 + y2_half^2 + y2_half^4 + 1
  return y
end
