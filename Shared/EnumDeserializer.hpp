#pragma once

#define BEGIN_ENUM_DESERIALIZER(x, type)\
std::string Deserialize##x(type nValue)\
{\
	std::string sDeserializedValue;

#define ENUM_DESERIALIZER_ENTRY(x)\
	if (nValue == x)\
	{\
		return #x;\
	}

#define END_ENUM_DESERIALIZER()\
	return "";\
}