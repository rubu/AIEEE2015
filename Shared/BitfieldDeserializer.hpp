#pragma once

#define BEGIN_BITFIELD_DESERIALIZER(x, type)\
std::string Deserialize##x(type nValue)\
{\
	std::string sDeserializedValue;

#define BITFIELD_DESERIALIZER_ENTRY(x)\
	if (nValue & x)\
	{\
		if (sDeserializedValue.size() > 0)\
		{\
			sDeserializedValue.append("|");\
		}\
		sDeserializedValue.append(#x);\
	}

#define END_BITFIELD_DESERIALIZER()\
	return sDeserializedValue;\
}