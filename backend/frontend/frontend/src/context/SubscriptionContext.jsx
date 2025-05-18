import React, { createContext, useState, useEffect, useContext } from 'react';
import { useUser } from '../hooks/useUser';

// Create and export the context
export const SubscriptionContext = createContext(null);

export const useSubscription = () => useContext(SubscriptionContext);

export const SUBSCRIPTION_TIERS = {
  FREE: 'free',
  PREMIUM: 'premium',
  ENTERPRISE: 'enterprise'
};

export const FEATURE_ACCESS = {
  'document-processing': {
    free: {
      allowed: true,
      limit: 5, // 5 documents per month
      message: 'Free users can process up to 5 documents per month'
    },
    premium: {
      allowed: true,
      limit: 100,
      message: 'Premium users can process up to 100 documents per month'
    },
    enterprise: {
      allowed: true,
      limit: -1, // unlimited
      message: 'Enterprise users have unlimited document processing'
    }
  },
  'custom-templates': {
    free: {
      allowed: false,
      message: 'Upgrade to Premium to create custom templates'
    },
    premium: {
      allowed: true,
      limit: 10,
      message: 'Premium users can create up to 10 custom templates'
    },
    enterprise: {
      allowed: true,
      limit: -1,
      message: 'Enterprise users can create unlimited custom templates'
    }
  },
  'advanced-formatting': {
    free: {
      allowed: false,
      message: 'Upgrade to Premium for advanced formatting options'
    },
    premium: {
      allowed: true,
      message: 'Access to all advanced formatting features'
    },
    enterprise: {
      allowed: true,
      message: 'Access to all advanced formatting features'
    }
  }
};

export const SubscriptionProvider = ({ children }) => {
  const { user } = useUser();
  const [currentTier, setCurrentTier] = useState(SUBSCRIPTION_TIERS.FREE);
  const [trialDays, setTrialDays] = useState(14);
  const [usage, setUsage] = useState({
    'document-processing': 0,
    'custom-templates': 0
  });

  // Trial period calculations
  useEffect(() => {
    if (user) {
      const joinDate = new Date(user.createdAt);
      const today = new Date();
      const diffTime = today.getTime() - joinDate.getTime();
      const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
      const trialLength = 14; // 14-day trial period
      const remainingDays = Math.max(0, trialLength - diffDays);
      setTrialDays(remainingDays);
    }
  }, [user]);

  const checkFeatureAccess = (feature) => {
    const featureConfig = FEATURE_ACCESS[feature]?.[currentTier];
    if (!featureConfig) return { allowed: false, message: 'Feature not available' };

    if (!featureConfig.allowed) {
      return { allowed: false, message: featureConfig.message };
    }

    if (featureConfig.limit !== -1 && usage[feature] >= featureConfig.limit) {
      return { 
        allowed: false, 
        message: `You've reached your ${feature} limit. Upgrade to increase your limit.`
      };
    }

    return { allowed: true, limit: featureConfig.limit, usage: usage[feature] };
  };

  const incrementUsage = (feature) => {
    setUsage(prev => ({
      ...prev,
      [feature]: (prev[feature] || 0) + 1
    }));
  };

  const value = {
    trialDays,
    currentTier,
    usage,
    checkFeatureAccess,
    incrementUsage,
    setCurrentTier,
  };

  return (
    <SubscriptionContext.Provider value={value}>
      {children}
    </SubscriptionContext.Provider>
  );
};