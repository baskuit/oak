#pragma once

template <typename Types>
struct Clamped : Types
{
	class Model : public Types::Model
	{
	public:
		using Types::Model::Model;
		void inference(Types::State &&state, Types::ModelOutput &output)
		{
			state.clamped = true;
			Types::Model::inference(std::move(state), output);
		}
	};
};
