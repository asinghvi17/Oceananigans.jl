#####
##### AbstractOperations with ZeroField and ConstantField
#####
##### Here we define a small number of operations with ZeroField
##### and ConstantField. Arbitrary AbstractOperations may not work.
#####

import Base: +, -, *, /, ==
using Oceananigans.Fields: ZeroField, ConstantField

# Binary operations
==(::ZeroField, ::ZeroField) = true

==(zf::ZeroField, cf::ConstantField) = 0 == cf.constant
==(cf::ConstantField, zf::ZeroField) = 0 == cf.constant
==(c1::ConstantField, c2::ConstantField) = c1.constant == c2.constant

+(a::ZeroField, b::AbstractField) = b
+(a::AbstractField, b::ZeroField) = a
+(a::ZeroField, b::Number) = ConstantField(b)
+(a::Number, b::ZeroField) = ConstantField(a)

-(a::ZeroField, b::AbstractField) = -b
-(a::AbstractField, b::ZeroField) = a
-(a::ZeroField, b::Number) = ConstantField(-b)
-(a::Number, b::ZeroField) = ConstantField(a)

*(a::ZeroField, b::AbstractField) = a
*(a::AbstractField, b::ZeroField) = b
*(a::ZeroField, b::Number) = a
*(a::Number, b::ZeroField) = b

/(a::ZeroField, b::AbstractField) = a
/(a::AbstractField, b::ZeroField) = ConstantField(convert(eltype(a), Inf))
/(a::ZeroField, b::Number) = a
/(a::Number, b::ZeroField) = ConstantField(a / convert(eltype(a), 0))

# for two ZeroField
for op in (:-, :+, :*)
    @eval begin
        function $op(z1::ZeroField{T1, N1}, z2::ZeroField{T2, N2}) where {T1, T2, N1, N2}
            T = Base.promote_type(T1, T2)
            N = max(N1, N2)
            return ZeroField{T, N}()
        end
    end
end

# Unary operations
-(a::ZeroField) = a
